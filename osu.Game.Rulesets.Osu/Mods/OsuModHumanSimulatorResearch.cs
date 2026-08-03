// Copyright (c) ppy Pty Ltd <contact@ppy.sh>. Licensed under the MIT Licence.
// See the LICENCE file in the repository root for full licence text.

using System;
using System.Diagnostics;
using System.IO;
using System.IO.Pipes;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using osu.Framework.Graphics.Sprites;
using osu.Framework.Localisation;
using osu.Framework.Logging;
using osu.Framework.Screens;
using osu.Game.Graphics;
using osu.Game.Graphics.Sprites;
using osu.Game.Rulesets.Mods;
using osu.Game.Rulesets.Osu.Objects;
using osu.Game.Rulesets.Osu.UI;
using osu.Game.Rulesets.UI;
using osu.Game.Screens.Play;
using osu.Game.Utils;
using osuTK;
using osuTK.Graphics;

namespace osu.Game.Rulesets.Osu.Mods
{
    /// <summary>
    /// Research-only handshake mod for the external human-movement simulator.
    /// It never creates replay frames and leaves normal player input handling active.
    /// </summary>
    public sealed class OsuModHumanSimulatorResearch : Mod, IResearchOnlyMod, IResearchGameplayStartHook,
                                                       IApplicableToDrawableRuleset<OsuHitObject>, IApplicableToPlayer, IApplicableToHUD,
                                                       IUpdatableByPlayfield, IDisposable
    {
        public override string Name => "Human Simulator Research";
        public override string Acronym => "HSR";
        public override ModType Type => ModType.Automation;
        public override LocalisableString Description => "Local synthetic research run. Requires the guarded external runner and cannot submit scores.";
        public override bool ValidForMultiplayer => false;
        public override bool ValidForMultiplayerAsFreeMod => false;

        public override Type[] IncompatibleMods => new[]
        {
            typeof(ModAutoplay),
            typeof(ModRelax),
            typeof(OsuModAutopilot),
            typeof(OsuModSpunOut),
        };

        private DrawableOsuRuleset? drawableRuleset;
        private NamedPipeClientStream? pipe;
        private StreamReader? reader;
        private StreamWriter? writer;
        private Player? player;
        private readonly SemaphoreSlim pipeWriteSemaphore = new(1, 1);
        private string? runToken;
        private long lastHeartbeatQpc;
        private int heartbeatWritePending;
        private volatile bool pipeFailed;
        private volatile bool runnerComplete;
        private volatile bool gameplayStartRequested;
        private volatile bool startMessageSent;
        private int startMessageWritePending;
        private bool gameplayClockObservedRunning;
        private bool abortRequested;
        private double clockRate = 1.0;

        public void ApplyToDrawableRuleset(DrawableRuleset<OsuHitObject> drawableRuleset)
            => this.drawableRuleset = (DrawableOsuRuleset)drawableRuleset;

        public void ApplyToHUD(HUDOverlay overlay)
        {
            overlay.TopLeftElements.Add(new OsuSpriteText
            {
                Text = "SYNTHETIC RESEARCH RUN",
                Font = OsuFont.GetFont(size: 18, weight: FontWeight.Bold),
                Colour = Color4.OrangeRed,
            });
        }

        public void ApplyToPlayer(Player player)
        {
            this.player = player;

            try
            {
                connectToRunner(player);
            }
            catch (Exception exception)
            {
                abortGameplay(exception, "Human simulator runner handshake failed; returning to song select.");
            }
        }

        private void connectToRunner(Player player)
        {
            if (drawableRuleset == null)
                throw new InvalidOperationException("Research drawable ruleset was not initialised.");

            string pipeName = Environment.GetEnvironmentVariable("HUMAN_SIM_PIPE")
                              ?? throw new InvalidOperationException("HUMAN_SIM_PIPE is missing. Launch this build through HumanSim.Runner.");
            runToken = Environment.GetEnvironmentVariable("HUMAN_SIM_RUN_TOKEN")
                       ?? throw new InvalidOperationException("HUMAN_SIM_RUN_TOKEN is missing. Launch this build through HumanSim.Runner.");

            string executablePath = Environment.ProcessPath
                                    ?? throw new InvalidOperationException("Unable to resolve the research client executable.");

            pipe = new NamedPipeClientStream(".", pipeName, PipeDirection.InOut, PipeOptions.Asynchronous);
            pipe.Connect(5000);
            reader = new StreamReader(pipe, leaveOpen: true);
            writer = new StreamWriter(pipe, leaveOpen: true) { AutoFlush = true };

            Vector2 origin = drawableRuleset.Playfield.GamefieldToScreenSpace(Vector2.Zero);
            Vector2 xAxis = drawableRuleset.Playfield.GamefieldToScreenSpace(new Vector2(512, 0));
            Vector2 yAxis = drawableRuleset.Playfield.GamefieldToScreenSpace(new Vector2(0, 384));
            clockRate = ModUtils.CalculateRateWithMods(player.GameplayState.Mods);

            var hello = new Handshake
            {
                RunToken = runToken,
                ProcessId = Environment.ProcessId,
                ExecutableSha256 = computeSha256(executablePath),
                BeatmapSha256 = player.GameplayState.Beatmap.BeatmapInfo.Hash,
                BeatmapMd5 = player.GameplayState.Beatmap.BeatmapInfo.MD5Hash,
                Mods = player.GameplayState.Mods.Where(m => m is not OsuModHumanSimulatorResearch).Select(m => m.Acronym).Order().ToArray(),
                ClockRate = clockRate,
                GameplayStartTimeMs = drawableRuleset.GameplayStartTime / clockRate,
                QpcFrequency = Stopwatch.Frequency,
                PlayfieldOrigin = new[] { origin.X, origin.Y },
                PlayfieldXAxis = new[] { xAxis.X, xAxis.Y },
                PlayfieldYAxis = new[] { yAxis.X, yAxis.Y },
            };

            writer.WriteLine(JsonSerializer.Serialize(hello));
            // Automatic mode decodes the exact selected difficulty through lazer and
            // generates its trace before acknowledging. Keep gameplay stopped until
            // the guarded runner proves that the matching trace is ready.
            string response = reader.ReadLineAsync().WaitAsync(TimeSpan.FromSeconds(180)).GetAwaiter().GetResult()
                              ?? throw new InvalidOperationException("Research runner disconnected before acknowledging the handshake.");
            var acknowledgement = JsonSerializer.Deserialize<Acknowledgement>(response)
                                  ?? throw new InvalidOperationException("Research runner returned a malformed acknowledgement.");

            if (!acknowledgement.Accepted || acknowledgement.RunToken != runToken)
                throw new InvalidOperationException($"Research runner rejected this client: {acknowledgement.Reason ?? "unknown reason"}");

            _ = Task.Run(() =>
            {
                try
                {
                    string message = reader.ReadLine() ?? throw new IOException("Research runner disconnected.");
                    using JsonDocument document = JsonDocument.Parse(message);
                    runnerComplete = document.RootElement.GetProperty("kind").GetString() == "complete"
                                     && document.RootElement.GetProperty("run_token").GetString() == runToken;
                    if (!runnerComplete)
                        pipeFailed = true;
                }
                catch
                {
                    if (!runnerComplete)
                        pipeFailed = true;
                }
            });
        }

        public bool OnGameplayStarting()
        {
            if (abortRequested)
                return false;

            if (writer == null || runToken == null)
            {
                abortGameplay(new InvalidOperationException("Research runner handshake was not completed."),
                    "Human simulator runner handshake was not completed; returning to song select.");
                return false;
            }

            long startQpc = Stopwatch.GetTimestamp();
            Interlocked.Exchange(ref lastHeartbeatQpc, startQpc);
            gameplayStartRequested = true;
            return true;
        }

        public void Update(Playfield playfield)
        {
            if (pipeFailed)
            {
                abortGameplay(new IOException("Research runner pipe disconnected."), "Human simulator runner pipe disconnected; returning to song select.");
                return;
            }
            if (gameplayStartRequested && !startMessageSent && playfield.Clock.IsRunning
                && Interlocked.CompareExchange(ref startMessageWritePending, 1, 0) == 0)
            {
                Vector2 origin = playfield.GamefieldToScreenSpace(Vector2.Zero);
                Vector2 xAxis = playfield.GamefieldToScreenSpace(new Vector2(512, 0));
                Vector2 yAxis = playfield.GamefieldToScreenSpace(new Vector2(0, 384));
                long startSyncQpc = Stopwatch.GetTimestamp();

                _ = sendStartMessageAsync(new StartMessage
                {
                    RunToken = runToken!,
                    Qpc = startSyncQpc,
                    // MapExporter and the generated trace use effective time
                    // after DT/HT transforms. The live gameplay clock advances
                    // in raw beatmap time at the active mod rate, so normalise
                    // every clock sample into the same effective-time domain.
                    GameplayClockTimeMs = playfield.Clock.CurrentTime / clockRate,
                    PlayfieldOrigin = new[] { origin.X, origin.Y },
                    PlayfieldXAxis = new[] { xAxis.X, xAxis.Y },
                    PlayfieldYAxis = new[] { yAxis.X, yAxis.Y },
                }, startSyncQpc);
            }

            if (startMessageSent && playfield.Clock.IsRunning)
                gameplayClockObservedRunning = true;
            else if (gameplayClockObservedRunning && !playfield.Clock.IsRunning)
            {
                abortGameplay(new InvalidOperationException("Research gameplay clock paused."), "Human simulator gameplay clock paused; returning to song select.");
                return;
            }
            if (runnerComplete || writer == null || runToken == null)
                return;
            long now = Stopwatch.GetTimestamp();

            if (!startMessageSent)
            {
                if ((now - Interlocked.Read(ref lastHeartbeatQpc)) * 1000.0 / Stopwatch.Frequency > 2000)
                    abortGameplay(new TimeoutException("Research gameplay-start message timed out."),
                        "Human simulator runner did not accept the gameplay start; returning to song select.");
                return;
            }

            // 50 ms heartbeats let the runner's clock model track the gameplay
            // clock tightly (and correct any drift) instead of only sampling it
            // four times per second.
            if ((now - Interlocked.Read(ref lastHeartbeatQpc)) * 1000.0 / Stopwatch.Frequency < 50)
                return;
            if (Interlocked.CompareExchange(ref heartbeatWritePending, 1, 0) != 0)
                return;

            _ = sendHeartbeatAsync(now, playfield.Clock.CurrentTime / clockRate, runToken);
        }

        private async Task sendStartMessageAsync(StartMessage message, long now)
        {
            try
            {
                await writeMessageAsync(JsonSerializer.Serialize(message), () =>
                {
                    Interlocked.Exchange(ref lastHeartbeatQpc, now);
                    startMessageSent = true;
                }).ConfigureAwait(false);
            }
            finally
            {
                Interlocked.Exchange(ref startMessageWritePending, 0);
            }
        }

        private async Task sendHeartbeatAsync(long now, double gameplayClockTimeMs, string token)
        {
            try
            {
                await writeMessageAsync(JsonSerializer.Serialize(new
                    {
                        kind = "heartbeat",
                        run_token = token,
                        qpc = now,
                        gameplay_clock_time_ms = gameplayClockTimeMs,
                    }),
                    () => Interlocked.Exchange(ref lastHeartbeatQpc, now)).ConfigureAwait(false);
            }
            finally
            {
                Interlocked.Exchange(ref heartbeatWritePending, 0);
            }
        }

        private async Task writeMessageAsync(string message, Action? onSuccess = null)
        {
            try
            {
                await pipeWriteSemaphore.WaitAsync().ConfigureAwait(false);
                try
                {
                    if (writer == null)
                        throw new IOException("Research runner pipe is unavailable.");

                    await writer.WriteLineAsync(message).WaitAsync(TimeSpan.FromSeconds(2)).ConfigureAwait(false);
                }
                finally
                {
                    pipeWriteSemaphore.Release();
                }

                onSuccess?.Invoke();
            }
            catch (Exception exception)
            {
                pipeFailed = true;
                Logger.Error(exception, "Asynchronous human simulator pipe write failed.");
            }
        }

        private void abortGameplay(Exception exception, string message)
        {
            if (abortRequested)
                return;

            abortRequested = true;
            pipeFailed = true;
            Logger.Error(exception, message);
            player?.Exit();
        }

        public void Dispose()
        {
            writer?.Dispose();
            reader?.Dispose();
            pipe?.Dispose();
            writer = null;
            reader = null;
            pipe = null;
        }

        private static string computeSha256(string path)
        {
            using FileStream stream = File.OpenRead(path);
            return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
        }

        private sealed class Handshake
        {
            [JsonPropertyName("protocol_version")]
            public int ProtocolVersion { get; init; } = 1;

            [JsonPropertyName("kind")]
            public string Kind { get; init; } = "hello";

            [JsonPropertyName("run_token")]
            public required string RunToken { get; init; }

            [JsonPropertyName("process_id")]
            public int ProcessId { get; init; }

            [JsonPropertyName("executable_sha256")]
            public required string ExecutableSha256 { get; init; }

            [JsonPropertyName("beatmap_sha256")]
            public required string BeatmapSha256 { get; init; }

            [JsonPropertyName("beatmap_md5")]
            public required string BeatmapMd5 { get; init; }

            [JsonPropertyName("mods")]
            public required string[] Mods { get; init; }

            [JsonPropertyName("clock_rate")]
            public double ClockRate { get; init; }

            [JsonPropertyName("gameplay_start_time_ms")]
            public double GameplayStartTimeMs { get; init; }

            [JsonPropertyName("qpc_frequency")]
            public long QpcFrequency { get; init; }

            [JsonPropertyName("playfield_origin")]
            public required float[] PlayfieldOrigin { get; init; }

            [JsonPropertyName("playfield_x_axis")]
            public required float[] PlayfieldXAxis { get; init; }

            [JsonPropertyName("playfield_y_axis")]
            public required float[] PlayfieldYAxis { get; init; }
        }

        private sealed class Acknowledgement
        {
            [JsonPropertyName("accepted")]
            public bool Accepted { get; init; }

            [JsonPropertyName("run_token")]
            public string? RunToken { get; init; }

            [JsonPropertyName("reason")]
            public string? Reason { get; init; }
        }

        private sealed class StartMessage
        {
            [JsonPropertyName("protocol_version")]
            public int ProtocolVersion { get; init; } = 1;

            [JsonPropertyName("kind")]
            public string Kind { get; init; } = "start";

            [JsonPropertyName("run_token")]
            public required string RunToken { get; init; }

            [JsonPropertyName("qpc")]
            public long Qpc { get; init; }

            [JsonPropertyName("gameplay_clock_time_ms")]
            public double GameplayClockTimeMs { get; init; }

            [JsonPropertyName("playfield_origin")]
            public required float[] PlayfieldOrigin { get; init; }

            [JsonPropertyName("playfield_x_axis")]
            public required float[] PlayfieldXAxis { get; init; }

            [JsonPropertyName("playfield_y_axis")]
            public required float[] PlayfieldYAxis { get; init; }
        }
    }
}
