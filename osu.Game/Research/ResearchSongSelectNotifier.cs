// Copyright (c) ppy Pty Ltd <contact@ppy.sh>. Licensed under the MIT Licence.
// See the LICENCE file in the repository root for full licence text.

using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Pipes;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using osu.Framework.Bindables;
using osu.Framework.Graphics;
using osu.Framework.Logging;
using osu.Game.Beatmaps;
using osu.Game.Rulesets.Mods;
using osu.Game.Utils;

namespace osu.Game.Research
{
    /// <summary>
    /// Research-only song-select notifier.
    ///
    /// When this build is launched by <c>HumanSim.Runner</c> in automatic mode,
    /// the runner publishes the <c>HUMAN_SIM_SELECTION_PIPE</c> environment
    /// variable. This component then reports every selected beatmap and the
    /// currently enabled research mods over that pipe, allowing the runner to
    /// parse and pre-plan a trace while the user is still browsing song select,
    /// before gameplay is entered.
    /// </summary>
    public partial class ResearchSongSelectNotifier : Component
    {
        private const int protocol_version = 1;
        private const double send_debounce_ms = 250;
        private const int connect_timeout_ms = 5000;
        private const int write_timeout_ms = 2000;

        private static readonly string[] supported_mods = { "HD", "HR", "DT", "HT", "FL" };

        private readonly Bindable<WorkingBeatmap> beatmap;
        private readonly Bindable<IReadOnlyList<Mod>> mods;
        private readonly string pipeName;
        private readonly string runToken;
        private readonly SemaphoreSlim writeSemaphore = new(1, 1);
        private readonly TaskCompletionSource<bool> connected = new(TaskCreationOptions.RunContinuationsAsynchronously);

        private NamedPipeClientStream? pipe;
        private StreamWriter? writer;
        private volatile bool failed;
        private bool sendScheduled;
        private double debounceRemainingMs;

        public ResearchSongSelectNotifier(Bindable<WorkingBeatmap> beatmap, Bindable<IReadOnlyList<Mod>> mods)
        {
            this.beatmap = beatmap;
            this.mods = mods;
            pipeName = Environment.GetEnvironmentVariable("HUMAN_SIM_SELECTION_PIPE") ?? string.Empty;
            runToken = Environment.GetEnvironmentVariable("HUMAN_SIM_RUN_TOKEN") ?? string.Empty;

            if (string.IsNullOrEmpty(pipeName) || string.IsNullOrEmpty(runToken))
                failed = true;
        }

        protected override void LoadComplete()
        {
            base.LoadComplete();

            if (failed)
                return;

            beatmap.BindValueChanged(_ => scheduleSend());
            mods.BindValueChanged(_ => scheduleSend());
            scheduleSend();

            _ = connectAsync();
        }

        protected override void Update()
        {
            base.Update();

            if (failed || !sendScheduled)
                return;

            debounceRemainingMs -= Clock.ElapsedFrameTime;
            if (debounceRemainingMs > 0)
                return;

            sendScheduled = false;
            _ = sendSelectionAsync();
        }

        private void scheduleSend()
        {
            if (failed)
                return;

            sendScheduled = true;
            debounceRemainingMs = send_debounce_ms;
        }

        private async Task connectAsync()
        {
            try
            {
                pipe = new NamedPipeClientStream(".", pipeName, PipeDirection.InOut, PipeOptions.Asynchronous);
                await pipe.ConnectAsync(connect_timeout_ms).ConfigureAwait(false);
                writer = new StreamWriter(pipe, leaveOpen: true) { AutoFlush = true };
                connected.TrySetResult(true);
            }
            catch (Exception exception)
            {
                connected.TrySetException(exception);
                failed = true;
                Logger.Log($"Human simulator selection notifier could not connect: {exception.Message}");
            }
        }

        private async Task sendSelectionAsync()
        {
            if (failed)
                return;

            try
            {
                await connected.Task.ConfigureAwait(false);

                BeatmapInfo? beatmapInfo = beatmap.Value?.BeatmapInfo;
                if (beatmapInfo == null || string.IsNullOrEmpty(beatmapInfo.Hash))
                    return;

                // Pre-planning only makes sense when the research mod is armed,
                // and only osu!standard mods supported by HumanSim.MapExporter
                // can be exported.
                if (mods.Value.All(m => m is not IResearchOnlyMod))
                    return;

                Mod[] researchExcluded = mods.Value.Where(m => m is not IResearchOnlyMod).ToArray();
                string[] acronyms = researchExcluded
                                    .Select(m => m.Acronym)
                                    .Where(acronym => supported_mods.Contains(acronym))
                                    .Order(StringComparer.Ordinal)
                                    .ToArray();
                double clockRate = ModUtils.CalculateRateWithMods(researchExcluded);

                string message = JsonSerializer.Serialize(new SelectionMessage
                {
                    RunToken = runToken,
                    BeatmapSha256 = beatmapInfo.Hash,
                    BeatmapMd5 = beatmapInfo.MD5Hash,
                    Mods = acronyms,
                    ClockRate = clockRate,
                });

                await writeSemaphore.WaitAsync().ConfigureAwait(false);
                try
                {
                    if (writer == null)
                        return;

                    await writer.WriteLineAsync(message).WaitAsync(TimeSpan.FromMilliseconds(write_timeout_ms)).ConfigureAwait(false);
                }
                finally
                {
                    writeSemaphore.Release();
                }
            }
            catch (Exception exception)
            {
                failed = true;
                Logger.Log($"Human simulator selection notifier stopped: {exception.Message}");
            }
        }

        protected override void Dispose(bool isDisposing)
        {
            beatmap.UnbindEvents();
            mods.UnbindEvents();

            // The runner process may already be gone when song select is torn
            // down; disposing a broken pipe throws synchronously inside the
            // async disposal queue, which surfaces as an unobserved error.
            try
            {
                writer?.Dispose();
            }
            catch (IOException)
            {
            }

            writer = null;

            try
            {
                pipe?.Dispose();
            }
            catch (IOException)
            {
            }

            pipe = null;
            writeSemaphore.Dispose();

            base.Dispose(isDisposing);
        }

        private sealed class SelectionMessage
        {
            [JsonPropertyName("protocol_version")]
            public int ProtocolVersion { get; init; } = protocol_version;

            [JsonPropertyName("kind")]
            public string Kind { get; init; } = "selected";

            [JsonPropertyName("run_token")]
            public required string RunToken { get; init; }

            [JsonPropertyName("beatmap_sha256")]
            public required string BeatmapSha256 { get; init; }

            [JsonPropertyName("beatmap_md5")]
            public required string BeatmapMd5 { get; init; }

            [JsonPropertyName("mods")]
            public required string[] Mods { get; init; }

            [JsonPropertyName("clock_rate")]
            public double ClockRate { get; init; }
        }
    }
}
