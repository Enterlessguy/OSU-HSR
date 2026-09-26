using System.Diagnostics;
using System.Globalization;
using System.IO.Compression;
using System.IO.Pipes;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using HumanSim.Transport;

namespace HumanSim.Runner;

internal static class Program
{
    private const int protocolVersion = 1;
    internal const string plannerVersion = "timing-sync-v2.17-coherent-test";
    private const string coherentAdapterVersion = "math-residual-lateral-gated-runtime-v1";

    private static readonly object prePlanLock = new();
    private static readonly Dictionary<string, Task<Trace>> prePlanTasks = new(StringComparer.Ordinal);
    private static readonly HashSet<string> supportedPlanMods = new(StringComparer.Ordinal) { "HD", "HR", "DT", "HT", "FL" };

    public static async Task<int> Main(string[] args)
    {
        if (args.Length == 1 && args[0].Equals("--print-identity", StringComparison.OrdinalIgnoreCase))
        {
            Console.WriteLine($"planner_version={plannerVersion}");
            return 0;
        }
        if (args.Length == 1 && args[0].Equals("--verify-runner-contracts", StringComparison.OrdinalIgnoreCase))
            return await verifyRunnerContracts().ConfigureAwait(false);
        if (args.Length == 1 && args[0].Equals("--verify-input-backend", StringComparison.OrdinalIgnoreCase))
        {
            try
            {
                if (OperatingSystem.IsWindows())
                    Console.WriteLine("backend=windows-sendinput;research-build-only=true");
                else if (OperatingSystem.IsLinux())
                {
                    Console.WriteLine("backend=research-client;display=x11-or-wayland;global-input=false;privileged-device-access=false");
                }
                else
                    throw new PlatformNotSupportedException("The HSR runner supports Windows and Linux research clients.");
                return 0;
            }
            catch (Exception exception)
            {
                Console.Error.WriteLine(exception.Message);
                return 1;
            }
        }
        StreamWriter? logWriter = null;
        try
        {
            Options options = Options.Parse(args);
            int verifyIndex = Array.IndexOf(args, "--verify-plan");
            bool clientPlayback = OperatingSystem.IsLinux() && !options.TimingOnly;
            if (!OperatingSystem.IsWindows() && !OperatingSystem.IsLinux())
                throw new PlatformNotSupportedException("The HSR runner supports Windows and Linux research clients.");
            if (verifyIndex >= 0)
            {
                string hash = args[verifyIndex + 1];
                string storage = options.OsuStoragePath ?? defaultOsuStoragePath();
                if (hash.Length != 64 || Convert.FromHexString(hash).Length != 32)
                    throw new InvalidDataException("--verify-plan requires a beatmap SHA-256");
                string map = Path.Combine(storage, hash[..1], hash[..2], hash);
                string md5 = Convert.ToHexString(MD5.HashData(File.ReadAllBytes(map))).ToLowerInvariant();
                Trace verified = await prepareAutomaticTraceFor(hash, md5, Array.Empty<string>(), 1.0, options, options.ClientPath).ConfigureAwait(false);
                Console.WriteLine(JsonSerializer.Serialize(new { verification = "compiled_runner_planning_only", frames = verified.Frames.Count, header = verified.Header }));
                return 0;
            }
            if (options.LogPath != null)
            {
                string logPath = Path.GetFullPath(options.LogPath);
                Directory.CreateDirectory(Path.GetDirectoryName(logPath)!);
                logWriter = new StreamWriter(logPath, append: false) { AutoFlush = true };
                Console.SetOut(new TeeTextWriter(Console.Out, logWriter));
                Console.SetError(new TeeTextWriter(Console.Error, logWriter));
            }
            Trace? trace = options.TracePath == null ? null : Trace.Load(options.TracePath);
            string executablePath = Path.GetFullPath(options.ClientPath);
            if (!File.Exists(executablePath)) throw new FileNotFoundException("Research client not found.", executablePath);

            string executableHash = sha256(executablePath);
            string pipeName = $"osu-human-sim-{Guid.NewGuid():N}";
            string token = Convert.ToHexString(RandomNumberGenerator.GetBytes(32)).ToLowerInvariant();
            string selectionPipeName = $"{pipeName}-selection";
            using var selectionPipe = new NamedPipeServerStream(selectionPipeName, PipeDirection.InOut, 1, PipeTransmissionMode.Byte,
                PipeOptions.Asynchronous | PipeOptions.CurrentUserOnly);
            using var process = Process.Start(new ProcessStartInfo(executablePath)
            {
                WorkingDirectory = Path.GetDirectoryName(executablePath)!,
                UseShellExecute = false,
                Environment =
                {
                    ["HUMAN_SIM_PIPE"] = pipeName,
                    ["HUMAN_SIM_RUN_TOKEN"] = token,
                    ["HUMAN_SIM_SELECTION_PIPE"] = selectionPipeName,
                    ["HUMAN_SIM_INPUT_BACKEND"] = clientPlayback ? "research-client" : "windows-sendinput",
                },
            }) ?? throw new InvalidOperationException("Research client did not start.");

            Console.WriteLine(options.AutoPlanMode == null
                ? $"Research client PID {process.Id} launched. Select the matching map and enable HSR."
                : $"Research client PID {process.Id} launched. Select any supported osu!standard map and enable HSR; its exact difficulty will be planned automatically. The runner stays alive between maps.");
            if (options.AutoPlanMode != null)
            {
                // Opportunistic pre-planning channel: the research client reports
                // the selected beatmap and mods while the user is still browsing
                // song select, so the trace can be ready before gameplay starts.
                _ = Task.Run(() => listenForSelectionsAsync(selectionPipe, token, options, executablePath, CancellationToken.None));
            }

            int servedRuns = 0;
            int completedRuns = 0;
            while (true)
            {
                try
                {
                    using var pipe = new NamedPipeServerStream(pipeName, PipeDirection.InOut, 1, PipeTransmissionMode.Byte,
                        PipeOptions.Asynchronous | PipeOptions.CurrentUserOnly);
                    // Idle waits must not be bounded by the per-map protocol
                    // timeout: the user may browse song select for a long time
                    // between maps. The runner only ends when the client process
                    // exits. The timeout token below is used solely for the
                    // handshake/start reads of an active connection.
                    Task pipeConnection = pipe.WaitForConnectionAsync();
                    Task clientExit = process.WaitForExitAsync();
                    if (await Task.WhenAny(pipeConnection, clientExit).ConfigureAwait(false) == clientExit)
                    {
                        if (servedRuns > 0)
                        {
                            Console.WriteLine($"Research client exited after {servedRuns} served run(s), {completedRuns} completed.");
                            return 0;
                        }

                        throw new InvalidOperationException($"Research client exited before connecting (code {process.ExitCode}).");
                    }

                    await pipeConnection.ConfigureAwait(false);

                    // The per-map protocol timeout starts only once a map has
                    // actually connected; the idle wait above is unbounded.
                    using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(options.TimeoutSeconds));
                    StreamReader? reader = null;
                    StreamWriter? writer = null;
                    string? payloadPath = null;
                    try
                    {
                        reader = new StreamReader(pipe, leaveOpen: true);
                        writer = new StreamWriter(pipe, leaveOpen: true) { AutoFlush = true };

                    Handshake hello = JsonSerializer.Deserialize<Handshake>(
                        await reader.ReadLineAsync(timeout.Token).ConfigureAwait(false) ?? throw new InvalidOperationException("Client disconnected before handshake."))
                        ?? throw new InvalidDataException("Malformed client handshake.");
                    string? rejection = validateClient(hello, process, token, executableHash);
                    if (rejection == null && options.AutoPlanMode != null)
                    {
                        try
                        {
                            trace = await prepareAutomaticTraceWithPrePlan(hello, options, executablePath).ConfigureAwait(false);
                        }
                        catch (Exception exception)
                        {
                            rejection = $"automatic map planning failed: {exception.Message}";
                        }
                    }

                    if (rejection == null && trace == null)
                        rejection = "no trace was supplied or generated";
                    if (rejection == null)
                        rejection = validateTrace(hello, trace!.Header);
                    string? payloadHash = null;
                    if (rejection == null && clientPlayback)
                    {
                        payloadPath = Path.Combine(Path.GetTempPath(), $"hsr-trace-{Guid.NewGuid():N}.json.gz");
                        var payload = new ResearchTraceData
                        {
                            RunToken = token, BeatmapSha256 = hello.BeatmapSha256, ClockRate = hello.ClockRate,
                            TimelineStartEffectiveMs = trace!.Header.TimelineStartEffectiveMs,
                            ExecutionMode = trace.Header.ExecutionMode,
                            LearnedSegments = trace.Header.ExecutionLearnedSegmentCount,
                            ChangedSamples = trace.Header.ExecutionChangedSampleCount,
                            FallbackSegments = trace.Header.ExecutionFallbackSegmentCount,
                            Frames = trace.Frames.Select(frame => new ResearchTraceFrame
                            { TimeUs = frame.TimeUs, X = frame.X, Y = frame.Y, K1 = frame.K1, K2 = frame.K2 }).ToList(),
                        };
                        payload.Validate(token, hello.BeatmapSha256, hello.ClockRate);
                        payloadHash = payload.Write(payloadPath);
                    }
                    await writer.WriteLineAsync(JsonSerializer.Serialize(new
                    {
                        accepted = rejection == null, run_token = token, reason = rejection,
                        input_backend = clientPlayback ? "research-client" : "windows-sendinput",
                        trace_path = payloadPath, trace_sha256 = payloadHash,
                    })).ConfigureAwait(false);
                    if (rejection != null)
                    {
                        Console.WriteLine($"Research client rejected: {rejection}");
                        continue;
                    }

                    servedRuns++;
                    Trace selectedTrace = trace!;

                    StartMessage start = JsonSerializer.Deserialize<StartMessage>(
                        await reader.ReadLineAsync(timeout.Token).ConfigureAwait(false) ?? throw new InvalidOperationException("Client disconnected before gameplay start."))
                        ?? throw new InvalidDataException("Malformed gameplay-start message.");
                    if (start.ProtocolVersion != protocolVersion || start.Kind != "start" || start.RunToken != token)
                        throw new InvalidDataException("Gameplay-start token or protocol mismatch.");
                    if (!double.IsFinite(start.GameplayClockTimeMs))
                        throw new InvalidDataException("Gameplay-start clock time is invalid.");
                    if (start.PlayfieldOrigin.Length != 2 || start.PlayfieldXAxis.Length != 2 || start.PlayfieldYAxis.Length != 2)
                        throw new InvalidDataException("Gameplay-start playfield transform is invalid.");

                    process.Refresh();
                    nint window = 0;
                    WindowGuard guard = default;
                    if (!options.TimingOnly && !clientPlayback)
                    {
                        window = process.MainWindowHandle;
                        if (window == 0) throw new InvalidOperationException("Research client has no main window.");
                        ensureForeground(process.Id);
                        guard = WindowGuard.Capture(window, process.Id);
                        Console.WriteLine($"Initial playfield transform: O=({hello.PlayfieldOrigin[0]:F2},{hello.PlayfieldOrigin[1]:F2}) X=({hello.PlayfieldXAxis[0]:F2},{hello.PlayfieldXAxis[1]:F2}) Y=({hello.PlayfieldYAxis[0]:F2},{hello.PlayfieldYAxis[1]:F2})");
                        Console.WriteLine($"Final playfield transform: O=({start.PlayfieldOrigin[0]:F2},{start.PlayfieldOrigin[1]:F2}) X=({start.PlayfieldXAxis[0]:F2},{start.PlayfieldXAxis[1]:F2}) Y=({start.PlayfieldYAxis[0]:F2},{start.PlayfieldYAxis[1]:F2})");
                        guard.ValidatePlayfield(start);
                        Console.WriteLine($"Research window: {guard.ClientRect.Width}x{guard.ClientRect.Height} desktop pixels ({guard.ClientRect.AspectRatio:F4}:1); virtual screen: {guard.VirtualScreen.Width}x{guard.VirtualScreen.Height}; scale {guard.Dpi / 96.0:F3}.");
                    }
                    var connection = new ConnectionMonitor(reader, token, start, !options.DisableClockFit);
                    var transformHolder = new TransformHolder(PlayfieldTransform.FromStart(start));
                    connection.AttachTransform(transformHolder, guard, options.TimingOnly || clientPlayback);
                    connection.Start();
                    Console.WriteLine($"Gameplay clock at launch: {start.GameplayClockTimeMs:F3} effective ms at {hello.ClockRate:F3}x; trace timeline starts at {selectedTrace.Header.TimelineStartEffectiveMs:F3} ms.");
                    if (clientPlayback)
                        await monitorClientPlayback(selectedTrace, process, connection).ConfigureAwait(false);
                    else
                        execute(selectedTrace, hello, start, process, window, guard, options, connection, transformHolder);
                    // Best-effort completion notification. If the client has
                    // already dropped the connection (user quit, map ended,
                    // harness closed), a completed run must not fail.
                    try
                    {
                        await writer.WriteLineAsync(JsonSerializer.Serialize(new { kind = "complete", run_token = token })).ConfigureAwait(false);
                    }
                    catch (IOException exception)
                    {
                        Console.WriteLine($"Completion notification failed: {exception.Message}");
                    }
                    Console.WriteLine("Synthetic research trace completed.");
                    completedRuns++;

                    // Keep the pipe open until the client ends gameplay and
                    // disconnects. Closing it right after "complete" makes the
                    // client's next heartbeat hit a broken pipe and abort the
                    // map's outro/results even though the trace finished.
                    for (int i = 0; i < 300 && !connection.IsDisconnected; i++)
                        await Task.Delay(50).ConfigureAwait(false);

                    if (options.AutoPlanMode == null)
                        return 0; // fixed-trace mode replays a single map
                    }
                    finally
                    {
                        if (payloadPath != null) File.Delete(payloadPath);
                        try { writer?.Dispose(); } catch (IOException) { }
                        try { reader?.Dispose(); } catch (IOException) { }
                    }
                }
                catch (OperationCanceledException)
                {
                    Console.Error.WriteLine("Timed out waiting for the research client handshake.");
                    Input.ReleaseAllKeys();
                    Console.WriteLine("Waiting for the next map...");
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine(ex.Message);
                    Input.ReleaseAllKeys();

                    if (options.AutoPlanMode == null)
                        return 1;

                    // A per-map abort (focus loss, window or DPI change, client
                    // disconnect, malformed trace, ...) must not end the session:
                    // release input, then listen for the next map's handshake.
                    Console.WriteLine("Waiting for the next map...");
                }
            }
        }
        catch (OperationCanceledException)
        {
            Console.Error.WriteLine("Timed out waiting for the research client handshake.");
            return 2;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex.Message);
            return 1;
        }
        finally
        {
            Input.ReleaseAllKeys();
            logWriter?.Dispose();
        }
    }

    private static string? validateClient(Handshake h, Process process, string token, string executableHash)
    {
        if (h.ProtocolVersion != protocolVersion || h.Kind != "hello") return "protocol mismatch";
        if (!fixedEquals(h.RunToken, token)) return "run-token mismatch";
        if (h.ProcessId != process.Id) return "client PID mismatch";
        if (!StringComparer.OrdinalIgnoreCase.Equals(h.ExecutableSha256, executableHash)) return "executable hash mismatch";
        if (!double.IsFinite(h.ClockRate) || h.ClockRate <= 0) return "invalid gameplay clock rate";
        if (h.QpcFrequency != Stopwatch.Frequency) return "high-resolution clock mismatch";
        if (h.PlayfieldOrigin.Length != 2 || h.PlayfieldXAxis.Length != 2 || h.PlayfieldYAxis.Length != 2) return "invalid playfield transform";
        return null;
    }

    private static async Task monitorClientPlayback(Trace trace, Process process, ConnectionMonitor connection)
    {
        Console.WriteLine($"Research client playback: {trace.Frames.Count} frames; learned segments={trace.Header.ExecutionLearnedSegmentCount}; changed samples={trace.Header.ExecutionChangedSampleCount}; fallback segments={trace.Header.ExecutionFallbackSegmentCount}.");
        long started = Stopwatch.GetTimestamp();
        double maximumSeconds = trace.Frames[^1].TimeUs / 1_000_000.0 + 120;
        while (!connection.ClientPlaybackComplete)
        {
            if (process.HasExited) throw new InvalidOperationException("Research client exited during synthetic playback.");
            connection.EnsureAlive();
            if (Stopwatch.GetElapsedTime(started).TotalSeconds > maximumSeconds)
                throw new TimeoutException("Research client did not finish the synthetic trace.");
            await Task.Delay(10).ConfigureAwait(false);
        }
        if (connection.ClientExpectedFrames != trace.Frames.Count || connection.ClientConsumedFrames != trace.Frames.Count)
            throw new InvalidDataException("Research client skipped trace frames; delivery parity failed.");
        Console.WriteLine($"Runtime telemetry: {JsonSerializer.Serialize(new
        {
            schema_version = 1, kind = "runtime_telemetry", status = "completed", timing_only = false,
            input_backend = "research-client", planned_frames = trace.Frames.Count,
            client_consumed_frames = connection.ClientConsumedFrames, delivered_frames = connection.ClientConsumedFrames,
            learned_segment_count = trace.Header.ExecutionLearnedSegmentCount,
            changed_sample_count = trace.Header.ExecutionChangedSampleCount,
            fallback_segment_count = trace.Header.ExecutionFallbackSegmentCount,
            heartbeat_count = connection.HeartbeatCount,
            heartbeat_max_gap_ms = connection.HeartbeatMaxGapMs,
            delivery_clock = "gameplay-frame-stability", os_dispatch_calibrated = false,
        })}");
    }

    private static string? validateTrace(Handshake h, TraceHeader t)
    {
        if (!StringComparer.OrdinalIgnoreCase.Equals(h.BeatmapSha256, t.BeatmapSha256)) return "beatmap hash mismatch";
        if (!string.IsNullOrEmpty(t.BeatmapMd5) && !StringComparer.OrdinalIgnoreCase.Equals(h.BeatmapMd5, t.BeatmapMd5)) return "beatmap MD5 mismatch";
        if (!h.Mods.Order().SequenceEqual(t.Mods.Order(), StringComparer.Ordinal)) return "active mods do not match trace";
        if (!double.IsFinite(t.ClockRate) || Math.Abs(h.ClockRate - t.ClockRate) > 1e-9) return "trace clock rate mismatch";
        return null;
    }

    private static bool fixedEquals(string a, string b)
    {
        try { return CryptographicOperations.FixedTimeEquals(Convert.FromHexString(a), Convert.FromHexString(b)); }
        catch (FormatException) { return false; }
    }

    private static async Task<Trace> prepareAutomaticTrace(Handshake hello, Options options, string executablePath)
        => await prepareAutomaticTraceFor(hello.BeatmapSha256, hello.BeatmapMd5, hello.Mods, hello.ClockRate, options, executablePath).ConfigureAwait(false);

    private static async Task<Trace> prepareAutomaticTraceFor(string beatmapSha256, string beatmapMd5, string[] mods, double clockRate, Options options, string executablePath)
    {
        string hash = beatmapSha256.ToLowerInvariant();
        try
        {
            if (hash.Length != 64 || Convert.FromHexString(hash).Length != 32)
                throw new InvalidDataException("selected beatmap has an invalid SHA-256 hash");
        }
        catch (FormatException)
        {
            throw new InvalidDataException("selected beatmap has an invalid SHA-256 hash");
        }

        string workspaceRoot = options.WorkspaceRoot == null
            ? findWorkspaceRoot(AppContext.BaseDirectory)
            : Path.GetFullPath(options.WorkspaceRoot);
        string storageRoot = options.OsuStoragePath == null
            ? defaultOsuStoragePath()
            : Path.GetFullPath(options.OsuStoragePath);
        string sourcePath = Path.Combine(storageRoot, hash[..1], hash[..2], hash);
        if (!File.Exists(sourcePath))
            throw new FileNotFoundException("selected beatmap content was not found in the local lazer store", sourcePath);
        if (!StringComparer.OrdinalIgnoreCase.Equals(sha256(sourcePath), hash))
            throw new InvalidDataException("selected beatmap content hash does not match the client handshake");

        // Ultra-dense novelty maps tap keys closer together than 500 Hz can
        // represent faithfully. Detect that from the source file before any
        // cache lookup, so dense maps always use the finest key timeline.
        Options effectiveOptions = options;
        if (options.SampleRateHz < 1000 && isUltraDenseOsuFile(sourcePath))
        {
            Console.WriteLine("Selected map has ultra-dense object spacing; planning at 1000 Hz for the finest key timeline.");
            effectiveOptions = options with { SampleRateHz = 1000 };
        }

        bool windows = OperatingSystem.IsWindows();
        string platformSuffix = windows ? ".exe" : string.Empty;
        string configuration = AppContext.BaseDirectory.Contains($"{Path.DirectorySeparatorChar}Release{Path.DirectorySeparatorChar}", StringComparison.OrdinalIgnoreCase)
            ? "Release"
            : "Debug";
        string exporter = Path.Combine(workspaceRoot, "research", "HumanSim.MapExporter", "bin", configuration, "net8.0", $"HumanSim.MapExporter{platformSuffix}");
        string planner = Path.Combine(workspaceRoot, "research", "human-sim", ".venv", windows ? "Scripts" : "bin",
            windows ? "human-sim.exe" : "human-sim");
        if (!File.Exists(exporter))
            throw new FileNotFoundException("HumanSim.MapExporter has not been built", exporter);
        if (!File.Exists(planner))
            throw new FileNotFoundException("the human-sim Python environment has not been created", planner);

        string modsKey = mods.Length == 0 ? "NM" : string.Join('-', mods.Order(StringComparer.Ordinal));
        string modeKey = options.AutoPlanMode == "perfect" ? "perfect" : "profile";
        string runKey = automaticRunKey(hash, mods, clockRate, effectiveOptions);
        string outputDirectory = options.OutputDirectory == null
            ? defaultAutoOutputDirectory(workspaceRoot)
            : Path.GetFullPath(options.OutputDirectory);
        Directory.CreateDirectory(outputDirectory);
        string mapPlanPath = Path.Combine(outputDirectory, $"{runKey}.map.ndjson.gz");
        string tracePath = Path.Combine(outputDirectory, $"{runKey}.trace.ndjson.gz");

        Trace? cached = tryLoadCachedTrace(tracePath, hash, beatmapMd5, mods, clockRate, effectiveOptions, workspaceRoot);
        if (cached != null)
        {
            Console.WriteLine($"Automatic trace cache hit: {tracePath}");
            return cached;
        }

        Console.WriteLine($"Automatically planning selected beatmap {hash} ({modsKey}, {modeKey}).");
        var exportArguments = new List<string>
        {
            "--map", sourcePath,
            "--output", mapPlanPath,
            "--mods", string.Join(',', mods),
            "--clock-rate", clockRate.ToString("R", CultureInfo.InvariantCulture),
        };
        await runTool("MapExporter", exporter, workspaceRoot, exportArguments, options.PlanningTimeoutSeconds).ConfigureAwait(false);

        async Task<string> runPlannerAtRate(int sampleRate, string targetTracePath)
        {
            var planArguments = new List<string>
            {
                "plan", mapPlanPath, targetTracePath,
                "--percentile", options.Percentile.ToString("0.###", CultureInfo.InvariantCulture),
                "--seed", options.Seed.ToString(CultureInfo.InvariantCulture),
                "--skill", (options.SkillLevel >= 0 ? options.SkillLevel : options.Percentile).ToString("0.###", CultureInfo.InvariantCulture),
                "--effort", options.EffortLevel.ToString("0.###", CultureInfo.InvariantCulture),
                "--sample-rate", sampleRate.ToString(CultureInfo.InvariantCulture),
                "--motion-mode", options.MotionMode,
                "--execution-mode", options.ExecutionMode,
                "--execution-blend", options.ExecutionBlend.ToString("R", CultureInfo.InvariantCulture),
                "--client-build", executablePath,
            };
            if (options.ExecutionModelPath != null)
            {
                planArguments.Add("--execution-model");
                planArguments.Add(options.ExecutionModelPath);
            }
            if (options.AutoPlanMode == "perfect")
                planArguments.Add("--perfect-baseline");
            await runTool("human-sim planner", planner, Path.GetDirectoryName(planner)!, planArguments, options.PlanningTimeoutSeconds,
                Path.Combine(workspaceRoot, "research", "human-sim", "src")).ConfigureAwait(false);
            return targetTracePath;
        }

        string plannedTracePath = await runPlannerAtRate(effectiveOptions.SampleRateHz, tracePath).ConfigureAwait(false);

        Trace trace = Trace.Load(plannedTracePath);
        Console.WriteLine($"Automatic trace ready: {trace.Frames.Count:N0} frames at {trace.Header.SampleRateHz} Hz from {mapPlanPath}");
        return trace;
    }

    private static async Task<Trace> prepareAutomaticTraceWithPrePlan(Handshake hello, Options options, string executablePath)
    {
        string key = automaticRunKey(hello.BeatmapSha256, hello.Mods, hello.ClockRate, options);
        Task<Trace>? prePlanned;
        lock (prePlanLock)
            prePlanTasks.TryGetValue(key, out prePlanned);

        if (prePlanned != null)
        {
            try
            {
                Trace trace = await prePlanned.WaitAsync(TimeSpan.FromSeconds(options.PlanningTimeoutSeconds)).ConfigureAwait(false);
                Console.WriteLine($"Pre-planned trace matched the selected difficulty ({key}).");
                return trace;
            }
            catch (Exception exception)
            {
                Console.WriteLine($"Pre-planned trace was unusable ({exception.Message}); planning fresh.");
            }
        }

        return await prepareAutomaticTrace(hello, options, executablePath).ConfigureAwait(false);
    }

    private static string automaticRunKey(string beatmapSha256, IEnumerable<string> mods, double clockRate, Options options)
    {
        string normalizedHash = beatmapSha256.ToLowerInvariant();
        string rateKey = clockRate.ToString("0.###", CultureInfo.InvariantCulture);
        string modsKey = !mods.Any() ? $"NM-r{rateKey}" : $"{string.Join('-', mods.Order(StringComparer.Ordinal))}-r{rateKey}";
        string modeKey = options.AutoPlanMode == "perfect" ? "perfect" : "profile";
        double effectiveSkill = options.SkillLevel >= 0 ? options.SkillLevel : options.Percentile;
        string modelKey = options.ExecutionModelPath == null ? "none" : sha256(options.ExecutionModelPath)[..12];
        return $"{normalizedHash[..12]}-{modsKey}-p{options.Percentile.ToString("0.###", CultureInfo.InvariantCulture)}-s{options.Seed}-sk{effectiveSkill.ToString("0.###", CultureInfo.InvariantCulture)}-ef{options.EffortLevel.ToString("0.###", CultureInfo.InvariantCulture)}-{options.SampleRateHz}hz-{modeKey}-ex{options.ExecutionMode}-xm{modelKey}";
    }

    private static bool isUltraDenseOsuFile(string path)
    {
        try
        {
            bool inObjects = false;
            double? previousTime = null;
            double minimumGap = double.MaxValue;
            foreach (string rawLine in File.ReadLines(path))
            {
                string line = rawLine.Trim();
                if (line.StartsWith("[HitObjects]", StringComparison.Ordinal))
                {
                    inObjects = true;
                    continue;
                }
                if (!inObjects || line.Length == 0 || line.StartsWith("//", StringComparison.Ordinal))
                    continue;
                if (line.StartsWith('['))
                    break;

                int firstComma = line.IndexOf(',');
                if (firstComma < 0) continue;
                int secondComma = line.IndexOf(',', firstComma + 1);
                if (secondComma < 0) continue;
                if (!double.TryParse(line.AsSpan(firstComma + 1, secondComma - firstComma - 1), NumberStyles.Float, CultureInfo.InvariantCulture, out double time))
                    continue;
                if (previousTime.HasValue)
                    minimumGap = Math.Min(minimumGap, time - previousTime.Value);
                previousTime = time;
            }
            return minimumGap < 8.0;
        }
        catch
        {
            return false;
        }
    }

    private static Trace? tryLoadCachedTrace(string tracePath, string beatmapSha256, string beatmapMd5, string[] mods, double clockRate, Options options, string workspaceRoot)
    {
        if (!File.Exists(tracePath))
            return null;

        try
        {
            Trace trace = Trace.Load(tracePath);
            TraceHeader header = trace.Header;
            if (!StringComparer.OrdinalIgnoreCase.Equals(header.BeatmapSha256, beatmapSha256)) return null;
            if (!string.IsNullOrEmpty(beatmapMd5) && !StringComparer.OrdinalIgnoreCase.Equals(header.BeatmapMd5, beatmapMd5)) return null;
            if (!header.Mods.Order(StringComparer.Ordinal).SequenceEqual(mods.Order(StringComparer.Ordinal))) return null;
            if (Math.Abs(header.ClockRate - clockRate) > 1e-9) return null;
            if (header.SampleRateHz != options.SampleRateHz) return null;
            if (header.DiagnosticPerfect != (options.AutoPlanMode == "perfect")) return null;
            if (!StringComparer.OrdinalIgnoreCase.Equals(header.ExecutionMode, options.ExecutionMode)) return null;
            if (!StringComparer.OrdinalIgnoreCase.Equals(header.PlannerVersion, plannerVersion)) return null;
            string expectedGitCommit = gitCommit(workspaceRoot);
            if (!String.Equals(expectedGitCommit, "unknown", StringComparison.OrdinalIgnoreCase)
                && !StringComparer.OrdinalIgnoreCase.Equals(header.GitCommit, expectedGitCommit))
                return null;

            string manifestPath = tracePath + ".manifest.json";
            if (!File.Exists(manifestPath)) return null;
            using JsonDocument manifest = JsonDocument.Parse(File.ReadAllBytes(manifestPath));
            if (!StringComparer.OrdinalIgnoreCase.Equals(manifest.RootElement.GetProperty("beatmap_sha256").GetString(), beatmapSha256))
                return null;
            if (!StringComparer.OrdinalIgnoreCase.Equals(manifest.RootElement.GetProperty("configuration_sha256").GetString(), canonicalConfiguration(options)))
                return null;
            if (!StringComparer.OrdinalIgnoreCase.Equals(manifest.RootElement.GetProperty("planner_version").GetString(), plannerVersion))
                return null;
            if (!StringComparer.OrdinalIgnoreCase.Equals(manifest.RootElement.GetProperty("git_commit").GetString(), header.GitCommit))
                return null;
            if (!StringComparer.OrdinalIgnoreCase.Equals(manifest.RootElement.GetProperty("build_identity").GetString(), header.BuildIdentity))
                return null;

            return trace;
        }
        catch (Exception exception)
        {
            Console.WriteLine($"Cached automatic trace {tracePath} rejected: {exception.Message}");
            return null;
        }
    }

    private static string canonicalConfiguration(Options options)
    {
        // Must match human_sim.cli._plan's deterministic sorted compact JSON.
        double effectiveSkill = options.SkillLevel >= 0 ? options.SkillLevel : options.Percentile;
        string modelHash = options.ExecutionModelPath == null ? "null" : $"\"{executionModelCanonicalSha(options.ExecutionModelPath)}\"";
        string adapter = options.ExecutionMode == "coherent" ? coherentAdapterVersion : "legacy";
        bool perfect = options.AutoPlanMode == "perfect" || options.MotionMode == "perfect";
        string json = $"{{\"effort_level\":{pythonFloat(options.EffortLevel)},\"execution_adapter\":\"{adapter}\",\"execution_blend\":{pythonFloat(options.ExecutionBlend)},\"execution_boundary_segments\":[],\"execution_model\":{modelHash},\"execution_mode\":\"{options.ExecutionMode}\",\"motion_mode\":\"{options.MotionMode}\",\"percentile\":{pythonFloat(options.Percentile)},\"perfect_baseline\":{(perfect ? "true" : "false")},\"planner_version\":\"{plannerVersion}\",\"sample_rate_hz\":{options.SampleRateHz},\"seed\":{options.Seed},\"skill_level\":{pythonFloat(effectiveSkill)}}}";
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(json))).ToLowerInvariant();
    }

    private static string executionModelCanonicalSha(string path)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllBytes(path));
        foreach (string property in new[] { "sha256", "model_sha256" })
        {
            if (document.RootElement.TryGetProperty(property, out JsonElement value)
                && value.ValueKind == JsonValueKind.String
                && !String.IsNullOrWhiteSpace(value.GetString()))
                return value.GetString()!;
        }
        throw new InvalidDataException($"execution model has no canonical hash: {path}");
    }

    private static string gitCommit(string workspaceRoot)
    {
        try
        {
            using Process process = Process.Start(new ProcessStartInfo("git")
            {
                WorkingDirectory = workspaceRoot,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
                Arguments = "rev-parse HEAD",
            }) ?? throw new InvalidOperationException("unable to start git");
            string output = process.StandardOutput.ReadToEnd().Trim();
            process.WaitForExit(2000);
            return process.ExitCode == 0 && output.Length > 0 ? output : "unknown";
        }
        catch
        {
            return "unknown";
        }
    }

    private static string pythonFloat(double value)
    {
        if (value == Math.Floor(value) && Math.Abs(value) < 1e15)
            return value.ToString("0.0", CultureInfo.InvariantCulture);
        return value.ToString("0.0########", CultureInfo.InvariantCulture);
    }

    private static async Task listenForSelectionsAsync(NamedPipeServerStream selectionPipe, string token, Options options, string executablePath, CancellationToken cancellationToken)
    {
        try
        {
            await selectionPipe.WaitForConnectionAsync(cancellationToken).ConfigureAwait(false);
            using var reader = new StreamReader(selectionPipe, leaveOpen: true);
            string? line;
            while ((line = await reader.ReadLineAsync(cancellationToken).ConfigureAwait(false)) != null)
            {
                if (string.IsNullOrWhiteSpace(line))
                    continue;

                SelectionMessage message;
                try
                {
                    message = JsonSerializer.Deserialize<SelectionMessage>(line) ?? throw new InvalidDataException("malformed selection message");
                }
                catch (Exception exception)
                {
                    Console.WriteLine($"[selection] {exception.Message}");
                    continue;
                }

                if (message.ProtocolVersion != protocolVersion || message.Kind != "selected" || !fixedEquals(message.RunToken, token))
                {
                    Console.WriteLine("[selection] rejected message (protocol or run-token mismatch)");
                    continue;
                }

                if (message.BeatmapSha256.Length != 64)
                {
                    Console.WriteLine("[selection] skipped invalid beatmap hash");
                    continue;
                }

                if (message.Mods.Any(mod => !supportedPlanMods.Contains(mod)))
                {
                    Console.WriteLine($"[selection] skipped unsupported mods: {string.Join(',', message.Mods)}");
                    continue;
                }

                string key = automaticRunKey(message.BeatmapSha256, message.Mods, message.ClockRate, options);
                lock (prePlanLock)
                {
                    if (prePlanTasks.ContainsKey(key))
                        continue;

                    Console.WriteLine($"[selection] pre-planning {key}...");
                    prePlanTasks[key] = Task.Run(() => prepareAutomaticTraceFor(message.BeatmapSha256, message.BeatmapMd5, message.Mods, message.ClockRate, options, executablePath));
                }
            }
        }
        catch (OperationCanceledException)
        {
        }
        catch (Exception exception)
        {
            Console.WriteLine($"[selection] listener stopped: {exception.Message}");
        }
    }

    private static string findWorkspaceRoot(string startPath)
    {
        for (DirectoryInfo? directory = new(Path.GetFullPath(startPath)); directory != null; directory = directory.Parent)
        {
            if (File.Exists(Path.Combine(directory.FullName, "global.json"))
                && Directory.Exists(Path.Combine(directory.FullName, "research", "HumanSim.MapExporter")))
                return directory.FullName;
        }
        throw new DirectoryNotFoundException("Unable to locate the human-simulator workspace root; pass --workspace-root explicitly.");
    }

    private static string defaultOsuStoragePath()
    {
        if (OperatingSystem.IsWindows())
            return Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "osu-development", "files");

        string? configuredDataHome = Environment.GetEnvironmentVariable("XDG_DATA_HOME");
        string dataHome = !string.IsNullOrWhiteSpace(configuredDataHome) && Path.IsPathFullyQualified(configuredDataHome)
            ? configuredDataHome
            : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".local", "share");
        return Path.Combine(dataHome, "osu-development", "files");
    }

    private static string defaultAutoOutputDirectory(string workspaceRoot)
    {
        if (OperatingSystem.IsWindows())
            return Path.Combine(workspaceRoot, "research", "human-sim", "output", "auto");

        string? configuredCacheHome = Environment.GetEnvironmentVariable("XDG_CACHE_HOME");
        string cacheHome = !string.IsNullOrWhiteSpace(configuredCacheHome) && Path.IsPathFullyQualified(configuredCacheHome)
            ? configuredCacheHome
            : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache");
        return Path.Combine(cacheHome, "intelligence-database-hsr", "auto");
    }

    private static async Task runTool(string label, string executable, string workingDirectory, IEnumerable<string> arguments, int timeoutSeconds, string? pythonSource = null)
    {
        bool assembly = Path.GetExtension(executable).Equals(".dll", StringComparison.OrdinalIgnoreCase);
        var startInfo = new ProcessStartInfo(assembly ? "dotnet" : executable)
        {
            WorkingDirectory = workingDirectory,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        if (assembly)
            startInfo.ArgumentList.Add(executable);
        foreach (string argument in arguments)
            startInfo.ArgumentList.Add(argument);
        if (pythonSource != null)
            startInfo.Environment["PYTHONPATH"] = pythonSource;

        using Process process = Process.Start(startInfo) ?? throw new InvalidOperationException($"Unable to start {label}.");
        Task<string> stdoutTask = process.StandardOutput.ReadToEndAsync();
        Task<string> stderrTask = process.StandardError.ReadToEndAsync();
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(timeoutSeconds));
        try
        {
            await process.WaitForExitAsync(timeout.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            if (!process.HasExited)
                process.Kill(entireProcessTree: true);
            throw new TimeoutException($"{label} exceeded the {timeoutSeconds}-second planning timeout.");
        }

        string stdout = (await stdoutTask.ConfigureAwait(false)).Trim();
        string stderr = (await stderrTask.ConfigureAwait(false)).Trim();
        if (stdout.Length > 0)
            Console.WriteLine($"[{label}] {stdout}");
        if (process.ExitCode != 0)
            throw new InvalidOperationException($"{label} exited with code {process.ExitCode}: {stderr}");
        if (stderr.Length > 0)
            Console.WriteLine($"[{label}] {stderr}");
    }

    private static void execute(Trace trace, Handshake h, StartMessage start, Process process, nint window, WindowGuard guard, Options options, ConnectionMonitor connection, TransformHolder transformHolder)
    {
        // osu-framework's screen-space coordinates are expressed in logical
        // pixels. SendInput absolute mouse coordinates are physical virtual-
        // desktop pixels, so applying the logical transform directly causes a
        // systematic DPI-dependent scale/offset error (for example, 1.25x at
        // 120 DPI). Convert the completed affine transform at the OS boundary.
        double logicalToPhysical = 96.0 / guard.Dpi;
        int effectiveCursorRate = Math.Min(options.CursorRateHz, trace.Header.SampleRateHz);
        IReadOnlyList<TraceFrame> dispatchFrames = selectDispatchFrames(trace, effectiveCursorRate);
        bool k1 = false, k2 = false;
        var dispatchLatenessUs = new LatencyStats();
        var keyDownLatenessUs = new LatencyStats();
        var sendDurationsUs = new LatencyStats();
        int deadlineCoalescedFrames = 0;
        int cadenceSkippedFrames = trace.Frames.Count - dispatchFrames.Count;
        bool executionCompleted = false;
        long nextGuardQpc = Stopwatch.GetTimestamp();
        long nextTelemetryUs = 5_000_000;
        ThreadPriority originalPriority = Thread.CurrentThread.Priority;
        bool timerResolutionActive = false;
        Console.WriteLine($"Executing a {trace.Header.SampleRateHz} Hz trace through a {effectiveCursorRate} Hz cursor cadence: {dispatchFrames.Count:N0} dispatch frames from {trace.Frames.Count:N0} trace frames (window DPI {guard.Dpi}, coordinate scale {logicalToPhysical:F4}, input lead {options.InputLeadMs:F3} ms).");
        try
        {
            // Windows default timer resolution (~15.6 ms) makes Thread.Sleep
            // overshoot badly. 1 ms resolution keeps the wait loop honest.
        timerResolutionActive = OperatingSystem.IsWindows() && !options.DisableTimerResolution && Native.timeBeginPeriod(1) == 0;

            // A real-time producer inside a normal-priority process can still be
            // preempted for up to a scheduling quantum (~15 ms), which is the
            // ~10-13 ms dispatch tail. High priority class reduces that window.
            if (OperatingSystem.IsWindows())
                Process.GetCurrentProcess().PriorityClass = ProcessPriorityClass.High;

            // This is a real-time producer. If Windows stalls it briefly, replaying
            // every obsolete 2 ms cursor sample makes the queue fall progressively
            // further behind. Run the scheduler at high priority and collapse only
            // overdue frames whose key state is unchanged. Key transitions are never
            // skipped, and the most recent cursor position is retained.
            if (OperatingSystem.IsWindows())
                Thread.CurrentThread.Priority = ThreadPriority.Highest;
            for (int i = 0; i < dispatchFrames.Count; i++)
            {
                TraceFrame f = dispatchFrames[i];
                // SendInput is consumed on a later osu-framework update. Keep
                // beatmap and trace timestamps canonical, and compensate for
                // that host-side delivery latency only at the OS boundary.
                double desiredClockTimeMs = trace.Header.TimelineStartEffectiveMs + f.TimeUs / 1000.0 - options.InputLeadMs;
                long targetQpc = connection.TargetQpc(desiredClockTimeMs);

                long nowQpc = Stopwatch.GetTimestamp();
                if (nowQpc > targetQpc)
                {
                    int latest = i;
                    long latestTargetQpc = targetQpc;
                    while (latest + 1 < dispatchFrames.Count)
                    {
                        TraceFrame next = dispatchFrames[latest + 1];
                        if (next.K1 != f.K1 || next.K2 != f.K2)
                            break;

                        double nextClockTimeMs = trace.Header.TimelineStartEffectiveMs + next.TimeUs / 1000.0 - options.InputLeadMs;
                        long nextTargetQpc = connection.TargetQpc(nextClockTimeMs);
                        if (nextTargetQpc > nowQpc)
                            break;

                        latest++;
                        latestTargetQpc = nextTargetQpc;
                    }

                    if (latest != i)
                    {
                        deadlineCoalescedFrames += latest - i;
                        i = latest;
                        f = dispatchFrames[i];
                        targetQpc = latestTargetQpc;
                    }
                }

                waitUntil(targetQpc);
                long dispatchQpc = Stopwatch.GetTimestamp();
                if (dispatchQpc >= nextGuardQpc)
                {
                    if (process.HasExited) throw new InvalidOperationException("Research client exited; run aborted.");
                    connection.EnsureAlive();
                    if (!options.TimingOnly)
                    {
                        ensureForeground(process.Id);
                        guard.Validate(window);
                    }
                    nextGuardQpc = dispatchQpc + Stopwatch.Frequency / 10;
                }

                double tx = f.X / 512.0, ty = f.Y / 384.0;
                PlayfieldTransform t = transformHolder.Current;
                // The mod's playfield transform is expressed relative to the
                // window's client area. SendInput needs absolute virtual-
                // desktop coordinates, so add the window's on-screen client
                // origin. Without this the whole path was shifted by the
                // window position (measured ~14 osu px in Y on this host),
                // which pushed rim landings outside the circle in-game even
                // though the planned trace was clean.
                double x = (t.OriginX + tx * (t.XAxisX - t.OriginX) + ty * (t.YAxisX - t.OriginX)) * logicalToPhysical + guard.ClientRect.Left;
                double y = (t.OriginY + tx * (t.XAxisY - t.OriginY) + ty * (t.YAxisY - t.OriginY)) * logicalToPhysical + guard.ClientRect.Top;
                double latenessUs = Math.Max(0, (dispatchQpc - targetQpc) * 1_000_000.0 / Stopwatch.Frequency);
                dispatchLatenessUs.Add(latenessUs);
                if ((f.K1 && !k1) || (f.K2 && !k2))
                    keyDownLatenessUs.Add(latenessUs);
                if (!options.TimingOnly)
                {
                    long sendStartedQpc = Stopwatch.GetTimestamp();
                    Input.SendFrame(x, y, f.K1, f.K2, ref k1, ref k2, options.LeftKey, options.RightKey);
                    sendDurationsUs.Add((Stopwatch.GetTimestamp() - sendStartedQpc) * 1_000_000.0 / Stopwatch.Frequency);
                }

                if (f.TimeUs >= nextTelemetryUs)
                {
                    ClockSyncSnapshot sync = connection.Snapshot();
                    Console.WriteLine($"Timing at {f.TimeUs / 1_000_000.0:F1}s: live-clock correction {sync.CorrectionMs:+0.000;-0.000;0.000} ms; dispatch p95 {dispatchLatenessUs.Percentile(0.95):F1} us, max {dispatchLatenessUs.Max:F1} us; key-down p95 {keyDownLatenessUs.Percentile(0.95):F1} us; {Input.BackendName} dispatch p95 {sendDurationsUs.Percentile(0.95):F1} us; deadline-coalesced {deadlineCoalescedFrames:N0} dispatch frames.");
                    do nextTelemetryUs += 5_000_000; while (nextTelemetryUs <= f.TimeUs);
                }
            }
            executionCompleted = true;
        }
        finally
        {
            if (OperatingSystem.IsWindows() && timerResolutionActive)
                Native.timeEndPeriod(1);
            if (OperatingSystem.IsWindows())
                Thread.CurrentThread.Priority = originalPriority;
            Input.ReleaseKey(options.LeftKey);
            Input.ReleaseKey(options.RightKey);
            if (dispatchLatenessUs.Count > 0)
                Console.WriteLine($"Dispatch lateness (us): all p50={dispatchLatenessUs.Percentile(0.50):F1}, p95={dispatchLatenessUs.Percentile(0.95):F1}, p99={dispatchLatenessUs.Percentile(0.99):F1}, max={dispatchLatenessUs.Max:F1}; key-down p95={keyDownLatenessUs.Percentile(0.95):F1}, max={keyDownLatenessUs.Max:F1}; {Input.BackendName} dispatch p95={sendDurationsUs.Percentile(0.95):F1}, max={sendDurationsUs.Max:F1}; cadence-skipped={cadenceSkippedFrames:N0}, deadline-coalesced={deadlineCoalescedFrames:N0}, delivered={dispatchLatenessUs.Count:N0}.");
            Console.WriteLine($"Runtime telemetry: {JsonSerializer.Serialize(new
            {
                schema_version = 1,
                kind = "runtime_telemetry",
                status = executionCompleted ? "completed" : "aborted",
                timing_only = options.TimingOnly,
                input_backend = Input.BackendId,
                dispatch_p50_us = dispatchLatenessUs.Percentile(0.50),
                dispatch_p95_us = dispatchLatenessUs.Percentile(0.95),
                dispatch_p99_us = dispatchLatenessUs.Percentile(0.99),
                dispatch_max_us = dispatchLatenessUs.Max,
                key_down_p95_us = keyDownLatenessUs.Percentile(0.95),
                key_down_max_us = keyDownLatenessUs.Max,
                dispatch_backend_p95_us = sendDurationsUs.Percentile(0.95),
                dispatch_backend_max_us = sendDurationsUs.Max,
                send_input_p95_us = sendDurationsUs.Percentile(0.95),
                send_input_max_us = sendDurationsUs.Max,
                cadence_skipped_frames = cadenceSkippedFrames,
                deadline_coalesced_frames = deadlineCoalescedFrames,
                delivered_frames = dispatchLatenessUs.Count,
                heartbeat_count = connection.HeartbeatCount,
                heartbeat_max_gap_ms = connection.HeartbeatMaxGapMs,
                clock_fit_valid = connection.FitValid,
            })}");
        }
    }

    private static IReadOnlyList<TraceFrame> selectDispatchFrames(Trace trace, int cursorRateHz)
    {
        if (cursorRateHz is < 60 or > 1000)
            throw new ArgumentOutOfRangeException(nameof(cursorRateHz), "Cursor rate must be between 60 and 1000 Hz.");
        if (cursorRateHz > trace.Header.SampleRateHz)
            throw new InvalidDataException("Cursor rate cannot exceed the trace sample rate.");

        double cursorPeriodUs = 1_000_000.0 / cursorRateHz;
        double nextCursorUs = trace.Frames[0].TimeUs;
        bool previousK1 = false, previousK2 = false;
        var selected = new List<TraceFrame>((int)Math.Ceiling(trace.Frames.Count * cursorRateHz / (double)trace.Header.SampleRateHz) + 1024);

        for (int i = 0; i < trace.Frames.Count; i++)
        {
            TraceFrame frame = trace.Frames[i];
            bool keyTransition = frame.K1 != previousK1 || frame.K2 != previousK2;
            bool cursorDue = frame.TimeUs + 0.5 >= nextCursorUs;
            if (cursorDue || keyTransition || i == trace.Frames.Count - 1)
                selected.Add(frame);

            if (cursorDue)
            {
                do nextCursorUs += cursorPeriodUs; while (nextCursorUs <= frame.TimeUs + 0.5);
            }
            previousK1 = frame.K1;
            previousK2 = frame.K2;
        }

        return selected;
    }

    private static void waitUntil(long target)
    {
        while (true)
        {
            long remaining = target - Stopwatch.GetTimestamp();
            if (remaining <= 0) return;
            double ms = remaining * 1000.0 / Stopwatch.Frequency;

            // Thread.Sleep overshoots even at 1 ms timer resolution, which
            // shows up directly as dispatch lateness on 240 Hz cursor cadence
            // (~4 ms between frames). Only sleep for longer horizons, then
            // busy-spin the short tail so the frame is dispatched on time.
            if (ms > 5)
                Thread.Sleep(Math.Max(1, (int)ms - 2));
            else
                Thread.SpinWait(512);
        }
    }

    private static void ensureForeground(int processId)
    {
        if (OperatingSystem.IsWindows())
        {
            Native.GetWindowThreadProcessId(Native.GetForegroundWindow(), out uint pid);
            if (pid != processId) throw new InvalidOperationException("Research client lost focus; run aborted.");
        }
        else
        {
            throw new PlatformNotSupportedException("Global input guards apply only to Windows; Linux uses private client playback.");
        }
    }

    private static async Task<int> verifyRunnerContracts()
    {
        try
        {
            string runToken = new('a', 64);
            string executableHash = new('b', 64);
            using Process process = Process.GetCurrentProcess();
            var handshake = new Handshake
            {
                ProtocolVersion = protocolVersion,
                Kind = "hello",
                RunToken = runToken,
                ProcessId = process.Id,
                ExecutableSha256 = executableHash,
                ClockRate = 1.0,
                QpcFrequency = Stopwatch.Frequency,
                PlayfieldOrigin = new[] { 0.0, 0.0 },
                PlayfieldXAxis = new[] { 512.0, 0.0 },
                PlayfieldYAxis = new[] { 0.0, 384.0 },
            };
            require(validateClient(handshake, process, runToken, executableHash) == null, "valid client handshake was rejected");
            require(validateClient(new Handshake { ProtocolVersion = 2, Kind = "hello", RunToken = runToken, ProcessId = process.Id, ExecutableSha256 = executableHash, ClockRate = 1, QpcFrequency = Stopwatch.Frequency, PlayfieldOrigin = new[] { 0.0, 0.0 }, PlayfieldXAxis = new[] { 1.0, 0.0 }, PlayfieldYAxis = new[] { 0.0, 1.0 } }, process, runToken, executableHash) == "protocol mismatch", "protocol mismatch was accepted");
            require(validateClient(handshake, process, new string('c', 64), executableHash) == "run-token mismatch", "mismatched token was accepted");
            require(validateClient(new Handshake { ProtocolVersion = protocolVersion, Kind = "hello", RunToken = runToken, ProcessId = process.Id + 1, ExecutableSha256 = executableHash, ClockRate = 1, QpcFrequency = Stopwatch.Frequency, PlayfieldOrigin = new[] { 0.0, 0.0 }, PlayfieldXAxis = new[] { 1.0, 0.0 }, PlayfieldYAxis = new[] { 0.0, 1.0 } }, process, runToken, executableHash) == "client PID mismatch", "mismatched PID was accepted");
            require(validateClient(new Handshake { ProtocolVersion = protocolVersion, Kind = "hello", RunToken = runToken, ProcessId = process.Id, ExecutableSha256 = new string('d', 64), ClockRate = 1, QpcFrequency = Stopwatch.Frequency, PlayfieldOrigin = new[] { 0.0, 0.0 }, PlayfieldXAxis = new[] { 1.0, 0.0 }, PlayfieldYAxis = new[] { 0.0, 1.0 } }, process, runToken, executableHash) == "executable hash mismatch", "mismatched executable hash was accepted");
            require(validateClient(new Handshake { ProtocolVersion = protocolVersion, Kind = "hello", RunToken = runToken, ProcessId = process.Id, ExecutableSha256 = executableHash, ClockRate = 1, QpcFrequency = Stopwatch.Frequency + 1, PlayfieldOrigin = new[] { 0.0, 0.0 }, PlayfieldXAxis = new[] { 1.0, 0.0 }, PlayfieldYAxis = new[] { 0.0, 1.0 } }, process, runToken, executableHash) == "high-resolution clock mismatch", "mismatched clock was accepted");

            var traceHeader = new TraceHeader
            {
                BeatmapSha256 = new string('e', 64),
                BeatmapMd5 = "0123456789abcdef0123456789abcdef",
                Mods = new[] { "HD" },
                ClockRate = 1,
            };
            var mapHandshake = new Handshake
            {
                BeatmapSha256 = traceHeader.BeatmapSha256,
                BeatmapMd5 = traceHeader.BeatmapMd5,
                Mods = new[] { "HD" },
                ClockRate = 1,
            };
            require(validateTrace(mapHandshake, traceHeader) == null, "valid trace/map identity was rejected");
            require(validateTrace(new Handshake { BeatmapSha256 = new string('f', 64), BeatmapMd5 = traceHeader.BeatmapMd5, Mods = new[] { "HD" }, ClockRate = 1 }, traceHeader) == "beatmap hash mismatch", "map-hash mismatch was accepted");
            require(validateTrace(new Handshake { BeatmapSha256 = traceHeader.BeatmapSha256, BeatmapMd5 = traceHeader.BeatmapMd5, Mods = Array.Empty<string>(), ClockRate = 1 }, traceHeader) == "active mods do not match trace", "mod mismatch was accepted");

            Trace scheduleTrace = new()
            {
                Header = new TraceHeader { SampleRateHz = 1000 },
                Frames = new[]
                {
                    new TraceFrame { TimeUs = 0, X = 0, Y = 0 },
                    new TraceFrame { TimeUs = 1000, X = 1, Y = 0 },
                    new TraceFrame { TimeUs = 1500, X = 2, Y = 0, K1 = true },
                    new TraceFrame { TimeUs = 2000, X = 3, Y = 0 },
                    new TraceFrame { TimeUs = 3000, X = 4, Y = 0, K2 = true },
                    new TraceFrame { TimeUs = 3500, X = 5, Y = 0, K2 = false },
                },
            };
            IReadOnlyList<TraceFrame> dispatched = selectDispatchFrames(scheduleTrace, 500);
            long[] dispatchedTimes = dispatched.Select(frame => frame.TimeUs).ToArray();
            require(dispatchedTimes.Contains(1500) && dispatchedTimes.Contains(2000), "cursor cadence dropped a key transition");
            require(dispatchedTimes.Contains(3000) && dispatchedTimes.Contains(3500), "second-key transitions were dropped");
            require(dispatchedTimes[^1] == 3500, "final frame was not retained");

            await verifyCurrentUserPipe(runToken).ConfigureAwait(false);
            Console.WriteLine("runner_contracts=passed; protocol=token,pid,hash,clock,map,mods; ipc=current-user-pipe; schedule=key-transitions,cadence,final-frame");
            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"runner contract verification failed: {exception.Message}");
            return 1;
        }
    }

    private static async Task verifyCurrentUserPipe(string runToken)
    {
        string pipeName = $"osu-hsr-contract-{Guid.NewGuid():N}";
        using var server = new NamedPipeServerStream(pipeName, PipeDirection.InOut, 1, PipeTransmissionMode.Byte,
            PipeOptions.Asynchronous | PipeOptions.CurrentUserOnly);
        Task<string?> clientExchange = Task.Run(async () =>
        {
            using var client = new NamedPipeClientStream(".", pipeName, PipeDirection.InOut, PipeOptions.Asynchronous);
            await client.ConnectAsync(5000).ConfigureAwait(false);
            using var writer = new StreamWriter(client, leaveOpen: true) { AutoFlush = true };
            using var reader = new StreamReader(client, leaveOpen: true);
            await writer.WriteLineAsync(JsonSerializer.Serialize(new { protocol_version = protocolVersion, kind = "hello", run_token = runToken })).ConfigureAwait(false);
            return await reader.ReadLineAsync().WaitAsync(TimeSpan.FromSeconds(5)).ConfigureAwait(false);
        });

        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        await server.WaitForConnectionAsync(timeout.Token).ConfigureAwait(false);
        using var serverReader = new StreamReader(server, leaveOpen: true);
        using var serverWriter = new StreamWriter(server, leaveOpen: true) { AutoFlush = true };
        using JsonDocument hello = JsonDocument.Parse(await serverReader.ReadLineAsync(timeout.Token).ConfigureAwait(false) ?? throw new InvalidDataException("pipe client sent no hello"));
        JsonElement root = hello.RootElement;
        require(root.GetProperty("protocol_version").GetInt32() == protocolVersion, "pipe protocol version mismatch");
        require(root.GetProperty("kind").GetString() == "hello", "pipe hello kind mismatch");
        require(fixedEquals(root.GetProperty("run_token").GetString() ?? string.Empty, runToken), "pipe token mismatch");
        await serverWriter.WriteLineAsync(JsonSerializer.Serialize(new Acknowledgement(true, runToken, null))).ConfigureAwait(false);
        string? acknowledgement = await clientExchange.WaitAsync(TimeSpan.FromSeconds(5)).ConfigureAwait(false);
        using JsonDocument response = JsonDocument.Parse(acknowledgement ?? throw new InvalidDataException("pipe client received no acknowledgement"));
        require(response.RootElement.GetProperty("accepted").GetBoolean(), "pipe acknowledgement was rejected");
        require(fixedEquals(response.RootElement.GetProperty("run_token").GetString() ?? string.Empty, runToken), "pipe acknowledgement token mismatch");
    }

    private static void require(bool condition, string message)
    {
        if (!condition)
            throw new InvalidOperationException(message);
    }

    private static string sha256(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }

}

internal sealed class PlayfieldTransform
{
    public readonly double OriginX, OriginY, XAxisX, XAxisY, YAxisX, YAxisY;

    public PlayfieldTransform(double originX, double originY, double xAxisX, double xAxisY, double yAxisX, double yAxisY)
    {
        OriginX = originX;
        OriginY = originY;
        XAxisX = xAxisX;
        XAxisY = xAxisY;
        YAxisX = yAxisX;
        YAxisY = yAxisY;
    }

    public static PlayfieldTransform FromStart(StartMessage start)
        => FromArrays(start.PlayfieldOrigin, start.PlayfieldXAxis, start.PlayfieldYAxis);

    public static PlayfieldTransform FromArrays(double[] origin, double[] xAxis, double[] yAxis)
        => new(origin[0], origin[1], xAxis[0], xAxis[1], yAxis[0], yAxis[1]);
}

internal sealed class TransformHolder
{
    private volatile PlayfieldTransform current;

    public TransformHolder(PlayfieldTransform initial) => current = initial;

    public PlayfieldTransform Current => current;

    public void Update(PlayfieldTransform transform) => current = transform;
}

internal sealed class ConnectionMonitor
{
    public bool ClientPlaybackComplete => Volatile.Read(ref clientPlaybackComplete);
    public int ClientConsumedFrames => Volatile.Read(ref clientConsumedFrames);
    public int ClientExpectedFrames => Volatile.Read(ref clientExpectedFrames);
    private bool clientPlaybackComplete;
    private int clientConsumedFrames;
    private int clientExpectedFrames;
    private const int max_samples = 128;
    private const int min_fit_samples = 8;

    private readonly StreamReader reader;
    private readonly string token;
    private readonly object clockLock = new();
    private readonly long initialQpc;
    private readonly double initialGameplayClockTimeMs;
    private readonly bool fitEnabled;
    private TransformHolder? transformHolder;
    private WindowGuard guard;
    private bool timingOnly;
    private long clockQpc;
    private double gameplayClockTimeMs;
    private long lastHeartbeat = Stopwatch.GetTimestamp();
    private long lastHeartbeatArrivalQpc;
    private long heartbeatCount;
    private long heartbeatMaxGapTicks;
    private volatile bool disconnected;

    private readonly long[] sampleQpc = new long[max_samples];
    private readonly double[] sampleClockMs = new double[max_samples];
    private int sampleCount;
    private int sampleNext;
    private long lastSampleQpc;
    private long fitOriginQpc;
    private bool fitValid;
    private double fitSlopeMsPerSecond = 1000.0;
    private double fitInterceptMs;

    public ConnectionMonitor(StreamReader reader, string token, StartMessage start, bool fitEnabled)
    {
        this.reader = reader;
        this.token = token;
        this.fitEnabled = fitEnabled;
        initialQpc = clockQpc = start.Qpc;
        initialGameplayClockTimeMs = gameplayClockTimeMs = start.GameplayClockTimeMs;
    }

    public void Start() => _ = Task.Run(() =>
    {
        try
        {
            string? line;
            while ((line = reader.ReadLine()) != null)
            {
                using JsonDocument message = JsonDocument.Parse(line);
                string kind = message.RootElement.GetProperty("kind").GetString() ?? "";
                string messageToken = message.RootElement.GetProperty("run_token").GetString() ?? "";
                if (messageToken != token)
                    throw new InvalidDataException("Research message token mismatch.");

                if (kind == "transform_update")
                {
                    double[] origin = readPoint(message.RootElement, "playfield_origin");
                    double[] xAxis = readPoint(message.RootElement, "playfield_x_axis");
                    double[] yAxis = readPoint(message.RootElement, "playfield_y_axis");
                    try
                    {
                        if (!timingOnly)
                            guard.ValidatePlayfield(origin, xAxis, yAxis);
                        transformHolder?.Update(PlayfieldTransform.FromArrays(origin, xAxis, yAxis));
                        Console.WriteLine($"Settled playfield transform: O=({origin[0]:F2},{origin[1]:F2}) X=({xAxis[0]:F2},{xAxis[1]:F2}) Y=({yAxis[0]:F2},{yAxis[1]:F2})");
                    }
                    catch (Exception exception)
                    {
                        // A transient/settling transform must not abort the
                        // run; keep the previous mapping and continue.
                        Console.WriteLine($"Ignoring unsettled playfield transform: {exception.Message}");
                    }
                    Interlocked.Exchange(ref lastHeartbeat, Stopwatch.GetTimestamp());
                    continue;
                }

                if (kind != "heartbeat")
                    throw new InvalidDataException("Malformed research message.");
                if (message.RootElement.TryGetProperty("client_consumed_frames", out JsonElement consumed))
                {
                    int count = consumed.GetInt32();
                    int expected = message.RootElement.GetProperty("client_expected_frames").GetInt32();
                    if (count < ClientConsumedFrames || count > expected || expected < 0 || expected > ResearchTraceData.MaximumFrames)
                        throw new InvalidDataException("Invalid research client frame counters.");
                    Volatile.Write(ref clientConsumedFrames, count);
                    Volatile.Write(ref clientExpectedFrames, expected);
                    Volatile.Write(ref clientPlaybackComplete, message.RootElement.GetProperty("client_playback_complete").GetBoolean());
                }
                long qpc = message.RootElement.GetProperty("qpc").GetInt64();
                double clockTime = message.RootElement.GetProperty("gameplay_clock_time_ms").GetDouble();
                if (qpc <= 0 || !double.IsFinite(clockTime))
                    throw new InvalidDataException("Research heartbeat clock sample is invalid.");
                long arrivalQpc = Stopwatch.GetTimestamp();
                long previousArrivalQpc = Interlocked.Exchange(ref lastHeartbeatArrivalQpc, arrivalQpc);
                if (previousArrivalQpc > 0)
                {
                    long gap = Math.Max(0, arrivalQpc - previousArrivalQpc);
                    long current;
                    do
                    {
                        current = Interlocked.Read(ref heartbeatMaxGapTicks);
                        if (gap <= current)
                            break;
                    }
                    while (Interlocked.CompareExchange(ref heartbeatMaxGapTicks, gap, current) != current);
                }
                Interlocked.Increment(ref heartbeatCount);
                lock (clockLock)
                {
                    clockQpc = qpc;
                    gameplayClockTimeMs = clockTime;

                    // Feed the drift-fitting model. Samples must be strictly
                    // increasing in QPC or the linear fit degenerates.
                    if (qpc > lastSampleQpc)
                    {
                        lastSampleQpc = qpc;
                        sampleQpc[sampleNext] = qpc;
                        sampleClockMs[sampleNext] = clockTime;
                        sampleNext = (sampleNext + 1) % max_samples;
                        if (sampleCount < max_samples)
                            sampleCount++;
                        recomputeFitLocked();
                    }
                }
                Interlocked.Exchange(ref lastHeartbeat, arrivalQpc);
            }
        }
        catch
        {
            disconnected = true;
        }
        disconnected = true;
    });

    public void AttachTransform(TransformHolder holder, WindowGuard windowGuard, bool timingOnly)
    {
        transformHolder = holder;
        guard = windowGuard;
        this.timingOnly = timingOnly;
    }

    private static double[] readPoint(JsonElement element, string property)
    {
        if (!element.TryGetProperty(property, out JsonElement point) || point.GetArrayLength() != 2)
            throw new InvalidDataException($"Research transform property {property} is invalid.");
        double[] values = new double[2];
        for (int i = 0; i < 2; i++)
        {
            values[i] = point[i].GetDouble();
            if (!double.IsFinite(values[i]))
                throw new InvalidDataException($"Research transform property {property} is not finite.");
        }
        return values;
    }

    public void EnsureAlive()
    {
        if (disconnected)
            throw new InvalidOperationException("Research client pipe disconnected; run aborted.");
        double ageSeconds = (Stopwatch.GetTimestamp() - Interlocked.Read(ref lastHeartbeat)) / (double)Stopwatch.Frequency;
        if (ageSeconds > 2)
            throw new InvalidOperationException("Research client heartbeat timed out; run aborted.");
    }

    public long TargetQpc(double desiredGameplayClockTimeMs)
    {
        lock (clockLock)
        {
            if (fitEnabled && fitValid)
            {
                // Anchor at the latest heartbeat so the extrapolation distance
                // stays within one heartbeat interval; the fitted slope alone
                // compensates clock drift (e.g. 1000.3 ms/s instead of 1000).
                double seconds = (desiredGameplayClockTimeMs - gameplayClockTimeMs) / fitSlopeMsPerSecond;
                return clockQpc + (long)Math.Round(seconds * Stopwatch.Frequency);
            }

            return clockQpc + (long)Math.Round((desiredGameplayClockTimeMs - gameplayClockTimeMs) * Stopwatch.Frequency / 1000.0);
        }
    }

    public bool IsDisconnected => disconnected;

    public long HeartbeatCount => Interlocked.Read(ref heartbeatCount);

    public double HeartbeatMaxGapMs => Interlocked.Read(ref heartbeatMaxGapTicks) * 1000.0 / Stopwatch.Frequency;

    public bool FitValid
    {
        get
        {
            lock (clockLock)
                return fitValid;
        }
    }

    public ClockSyncSnapshot Snapshot()
    {
        lock (clockLock)
        {
            double predictedClockTimeMs;
            if (fitEnabled && fitValid)
                predictedClockTimeMs = fitInterceptMs + fitSlopeMsPerSecond * (clockQpc - fitOriginQpc) / Stopwatch.Frequency;
            else
                predictedClockTimeMs = initialGameplayClockTimeMs + (clockQpc - initialQpc) * 1000.0 / Stopwatch.Frequency;
            return new ClockSyncSnapshot(clockQpc, gameplayClockTimeMs, gameplayClockTimeMs - predictedClockTimeMs);
        }
    }

    private void recomputeFitLocked()
    {
        if (sampleCount < min_fit_samples)
        {
            fitValid = false;
            return;
        }

        int start = (sampleNext - sampleCount + max_samples) % max_samples;
        long origin = sampleQpc[start];
        double sumX = 0, sumY = 0, sumXX = 0, sumXY = 0;
        for (int k = 0; k < sampleCount; k++)
        {
            int index = (start + k) % max_samples;
            double x = (sampleQpc[index] - origin) / (double)Stopwatch.Frequency;
            double y = sampleClockMs[index];
            sumX += x;
            sumY += y;
            sumXX += x * x;
            sumXY += x * y;
        }

        double meanX = sumX / sampleCount;
        double meanY = sumY / sampleCount;
        double variance = sumXX - sampleCount * meanX * meanX;
        if (variance < 1e-9)
        {
            fitValid = false;
            return;
        }

        double slope = (sumXY - sampleCount * meanX * meanY) / variance;
        if (slope < 500 || slope > 2000)
        {
            fitValid = false;
            return;
        }

        fitSlopeMsPerSecond = slope;
        fitInterceptMs = meanY - slope * meanX;
        fitOriginQpc = origin;
        fitValid = true;
    }
}

internal readonly record struct ClockSyncSnapshot(long Qpc, double GameplayClockTimeMs, double CorrectionMs);

internal sealed class LatencyStats
{
    private const double bucketWidthUs = 10.0;
    private const int bucketCount = 100_001;
    private readonly int[] buckets = new int[bucketCount];

    public long Count { get; private set; }
    public double Max { get; private set; }

    public void Add(double valueUs)
    {
        double value = Math.Max(0, valueUs);
        int bucket = Math.Min(bucketCount - 1, (int)(value / bucketWidthUs));
        buckets[bucket]++;
        Count++;
        Max = Math.Max(Max, value);
    }

    public double Percentile(double quantile)
    {
        if (Count == 0) return 0;
        long target = Math.Max(1, (long)Math.Ceiling(Count * Math.Clamp(quantile, 0, 1)));
        long cumulative = 0;
        for (int i = 0; i < buckets.Length; i++)
        {
            cumulative += buckets[i];
            if (cumulative >= target)
                return i * bucketWidthUs;
        }
        return Max;
    }
}

internal sealed record Options(
    string ClientPath,
    string? TracePath,
    string? AutoPlanMode,
    int TimeoutSeconds,
    int PlanningTimeoutSeconds,
    ushort LeftKey,
    ushort RightKey,
    double Percentile,
    int Seed,
    double SkillLevel,
    double EffortLevel,
    string MotionMode,
    string ExecutionMode,
    string? ExecutionModelPath,
    double ExecutionBlend,
    int SampleRateHz,
    int CursorRateHz,
    double InputLeadMs,
    string? WorkspaceRoot,
    string? OsuStoragePath,
    string? OutputDirectory,
    string? LogPath,
    bool TimingOnly,
    bool DisableTimerResolution,
    bool DisableClockFit)
{
    public static Options Parse(string[] args)
    {
        var values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        bool timingOnly = false;
        bool disableTimerResolution = false;
        bool disableClockFit = false;
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i].Equals("--timing-only", StringComparison.OrdinalIgnoreCase))
            {
                timingOnly = true;
                continue;
            }
            if (args[i].Equals("--disable-timer-resolution", StringComparison.OrdinalIgnoreCase))
            {
                disableTimerResolution = true;
                continue;
            }
            if (args[i].Equals("--disable-clock-fit", StringComparison.OrdinalIgnoreCase))
            {
                disableClockFit = true;
                continue;
            }
            if (i + 1 >= args.Length || !args[i].StartsWith("--", StringComparison.Ordinal))
                throw usage();
            values[args[i]] = args[i + 1];
            i++;
        }
        string? trace = values.GetValueOrDefault("--trace");
        string? autoPlan = values.GetValueOrDefault("--auto-plan")?.ToLowerInvariant();
        if ((trace == null) == (autoPlan == null))
            throw usage();
        if (autoPlan is not null and not ("perfect" or "profile"))
            throw new ArgumentException("--auto-plan must be 'perfect' or 'profile'.");
        int sampleRate = int.Parse(values.GetValueOrDefault("--sample-rate", "500"), CultureInfo.InvariantCulture);
        int cursorRate = int.Parse(values.GetValueOrDefault("--cursor-rate", "1000"), CultureInfo.InvariantCulture);
        double inputLeadMs = double.Parse(values.GetValueOrDefault("--input-lead-ms", "0"), CultureInfo.InvariantCulture);
        double skillLevel = double.Parse(values.GetValueOrDefault("--skill", "-1"), CultureInfo.InvariantCulture);
        double effortLevel = double.Parse(values.GetValueOrDefault("--effort", "100"), CultureInfo.InvariantCulture);
        string motionMode = values.GetValueOrDefault("--motion-mode", autoPlan ?? "profile").ToLowerInvariant();
        string executionMode = values.GetValueOrDefault("--execution-mode", "math-only").ToLowerInvariant();
        string? executionModel = values.GetValueOrDefault("--execution-model");
        double executionBlend = double.Parse(values.GetValueOrDefault("--execution-blend", executionMode == "math-only" ? "0" : "1"), CultureInfo.InvariantCulture);
        if (sampleRate is < 60 or > 1000) throw new ArgumentException("--sample-rate must be between 60 and 1000 Hz.");
        if (cursorRate is < 60 or > 1000) throw new ArgumentException("--cursor-rate must be between 60 and 1000 Hz.");
        if (!double.IsFinite(inputLeadMs) || inputLeadMs is < 0 or > 50)
            throw new ArgumentException("--input-lead-ms must be between 0 and 50 ms.");
        if (!double.IsFinite(skillLevel) || skillLevel is < -1 or > 100) throw new ArgumentException("--skill must be between 0 and 100.");
        if (!double.IsFinite(effortLevel) || effortLevel is < 0 or > 100) throw new ArgumentException("--effort must be between 0 and 100.");
        if (motionMode is not ("profile" or "perfect")) throw new ArgumentException("--motion-mode must be profile or perfect.");
        if (executionMode is not ("math-only" or "hybrid" or "coherent")) throw new ArgumentException("--execution-mode must be math-only, hybrid, or coherent.");
        if (!double.IsFinite(executionBlend) || executionBlend is < 0 or > 1) throw new ArgumentException("--execution-blend must be between 0 and 1.");
        if (executionMode == "coherent" && executionBlend <= 0) throw new ArgumentException("coherent execution requires a positive blend.");
        if (executionMode == "math-only" && (executionModel != null || executionBlend > 0)) throw new ArgumentException("math-only execution cannot receive a model or non-zero blend.");
        if (executionMode != "math-only" && executionBlend > 0 && executionModel == null) throw new ArgumentException($"{executionMode} execution requires --execution-model.");
        if (executionModel != null)
        {
            executionModel = Path.GetFullPath(executionModel);
            if (!File.Exists(executionModel)) throw new FileNotFoundException("execution model was not found", executionModel);
        }
        return new Options(
            values.GetValueOrDefault("--client") ?? throw new ArgumentException("--client is required"),
            trace,
            autoPlan,
            int.Parse(values.GetValueOrDefault("--timeout-seconds", "600"), CultureInfo.InvariantCulture),
            int.Parse(values.GetValueOrDefault("--planning-timeout-seconds", "180"), CultureInfo.InvariantCulture),
            parseKey(values.GetValueOrDefault("--left-key", "Z")),
            parseKey(values.GetValueOrDefault("--right-key", "X")),
            double.Parse(values.GetValueOrDefault("--percentile", "99.5"), CultureInfo.InvariantCulture),
            int.Parse(values.GetValueOrDefault("--seed", "42"), CultureInfo.InvariantCulture),
            skillLevel,
            effortLevel,
            motionMode,
            executionMode,
            executionModel,
            executionBlend,
            sampleRate,
            cursorRate,
            inputLeadMs,
            values.GetValueOrDefault("--workspace-root"),
            values.GetValueOrDefault("--osu-storage"),
            values.GetValueOrDefault("--output-directory"),
            values.GetValueOrDefault("--log-path"),
            timingOnly,
            disableTimerResolution,
            disableClockFit);
    }

    private static ArgumentException usage() => new(
        "Usage: HumanSim.Runner --client <osu.Desktop.exe> (--trace <trace.gz> | --auto-plan <perfect|profile>) [--motion-mode <profile|perfect>] [--execution-mode <math-only|hybrid|coherent>] [--execution-model <json>] [--execution-blend 0-1] [--percentile 99.5] [--seed 42] [--skill 0-100] [--effort 0-100] [--sample-rate 500] [--cursor-rate 1000] [--input-lead-ms 0] [--log-path <file>] [--timing-only] [--disable-timer-resolution] [--disable-clock-fit]");

    private static ushort parseKey(string value)
    {
        if (value.Length != 1 || !char.IsLetterOrDigit(value[0])) throw new ArgumentException("Keys must be one letter or digit.");
        return char.ToUpperInvariant(value[0]);
    }
}

internal sealed class Trace
{
    public required TraceHeader Header { get; init; }
    public required IReadOnlyList<TraceFrame> Frames { get; init; }

    public static Trace Load(string path)
    {
        using FileStream file = File.OpenRead(path);
        using Stream input = path.EndsWith(".gz", StringComparison.OrdinalIgnoreCase) ? new GZipStream(file, CompressionMode.Decompress) : file;
        using var reader = new StreamReader(input);
        var header = JsonSerializer.Deserialize<TraceHeader>(reader.ReadLine() ?? throw new InvalidDataException("Trace is empty."))
                     ?? throw new InvalidDataException("Malformed trace header.");
        if (header.Kind != "trace_header" || header.SchemaVersion != 1 || !header.Synthetic)
            throw new InvalidDataException("Runner accepts only schema-v1 visibly synthetic traces.");
        if (!StringComparer.OrdinalIgnoreCase.Equals(header.PlannerVersion, Program.plannerVersion)
            || String.IsNullOrWhiteSpace(header.GitCommit)
            || String.IsNullOrWhiteSpace(header.BuildIdentity))
            throw new InvalidDataException($"Trace planner identity is missing or stale; expected {Program.plannerVersion}.");
        if (header.SampleRateHz is < 60 or > 1000)
            throw new InvalidDataException("Trace sample rate must be between 60 and 1000 Hz.");
        if (header.ExecutionMode == "coherent" &&
            (header.ExecutionEffectiveMode != "coherent" || header.ExecutionFallback ||
             header.ExecutionLearnedSegmentCount <= 0 || header.ExecutionChangedSampleCount <= 0))
            throw new InvalidDataException("Coherent AI produced no delivered learned movement; refusing an all-math fallback trace.");

        var frames = new List<TraceFrame>();
        long previous = -1;
        TraceFrame? previousFrame = null;
        string? line;
        while ((line = reader.ReadLine()) != null)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            var frame = JsonSerializer.Deserialize<TraceFrame>(line) ?? throw new InvalidDataException("Malformed trace frame.");
            if (frame.TimeUs <= previous || !double.IsFinite(frame.X) || !double.IsFinite(frame.Y))
                throw new InvalidDataException("Trace timestamps must increase and coordinates must be finite.");
            if (frame.X is < -512 or > 1024 || frame.Y is < -512 or > 896)
                throw new InvalidDataException("Trace coordinate is outside the guarded playfield margin.");
            if (previousFrame != null)
            {
                double seconds = (frame.TimeUs - previousFrame.TimeUs) / 1_000_000.0;
                double dx = frame.X - previousFrame.X;
                double dy = frame.Y - previousFrame.Y;
                double speed = Math.Sqrt(dx * dx + dy * dy) / seconds;
                double speedLimit = header.DiagnosticPerfect ? 1_000_000 : 100_000;
                if (speed > speedLimit)
                    throw new InvalidDataException($"Trace contains an implausible cursor discontinuity ({speed:F1} osu! pixels/s, limit {speedLimit:F0}).");
            }
            previous = frame.TimeUs;
            previousFrame = frame;
            frames.Add(frame);
        }
        if (frames.Count == 0) throw new InvalidDataException("Trace contains no frames.");
        if (frames[^1].K1 || frames[^1].K2) throw new InvalidDataException("Trace ends with a held key.");
        return new Trace { Header = header, Frames = frames };
    }
}

internal sealed class TraceHeader
{
    [JsonPropertyName("kind")] public string Kind { get; init; } = "";
    [JsonPropertyName("schema_version")] public int SchemaVersion { get; init; }
    [JsonPropertyName("beatmap_sha256")] public string BeatmapSha256 { get; init; } = "";
    [JsonPropertyName("beatmap_md5")] public string BeatmapMd5 { get; init; } = "";
    [JsonPropertyName("mods")] public string[] Mods { get; init; } = Array.Empty<string>();
    [JsonPropertyName("sample_rate_hz")] public int SampleRateHz { get; init; }
    [JsonPropertyName("clock_rate")] public double ClockRate { get; init; }
    [JsonPropertyName("timeline_start_effective_ms")] public double TimelineStartEffectiveMs { get; init; }
    [JsonPropertyName("synthetic")] public bool Synthetic { get; init; }
    [JsonPropertyName("diagnostic_perfect")] public bool DiagnosticPerfect { get; init; }
    [JsonPropertyName("execution_mode")] public string ExecutionMode { get; init; } = "math-only";
    [JsonPropertyName("execution_effective_mode")] public string ExecutionEffectiveMode { get; init; } = "";
    [JsonPropertyName("execution_fallback")] public bool ExecutionFallback { get; init; }
    [JsonPropertyName("execution_learned_segment_count")] public int ExecutionLearnedSegmentCount { get; init; }
    [JsonPropertyName("execution_changed_sample_count")] public int ExecutionChangedSampleCount { get; init; }
    [JsonPropertyName("execution_fallback_segment_count")] public int ExecutionFallbackSegmentCount { get; init; }
    [JsonPropertyName("planner_version")] public string PlannerVersion { get; init; } = "";
    [JsonPropertyName("git_commit")] public string GitCommit { get; init; } = "";
    [JsonPropertyName("build_identity")] public string BuildIdentity { get; init; } = "";
}

internal sealed class TraceFrame
{
    [JsonPropertyName("time_us")] public long TimeUs { get; init; }
    [JsonPropertyName("x")] public double X { get; init; }
    [JsonPropertyName("y")] public double Y { get; init; }
    [JsonPropertyName("k1")] public bool K1 { get; init; }
    [JsonPropertyName("k2")] public bool K2 { get; init; }
}

/// <summary>
/// Mirrors console output into a log file while keeping it visible live.
/// </summary>
internal sealed class TeeTextWriter : TextWriter
{
    private readonly TextWriter primary;
    private readonly TextWriter secondary;

    public TeeTextWriter(TextWriter primary, TextWriter secondary)
    {
        this.primary = primary;
        this.secondary = secondary;
    }

    public override Encoding Encoding => primary.Encoding;

    public override void Write(char value)
    {
        primary.Write(value);
        secondary.Write(value);
    }

    public override void Write(string? value)
    {
        primary.Write(value);
        secondary.Write(value);
    }

    public override void WriteLine(string? value)
    {
        primary.WriteLine(value);
        secondary.WriteLine(value);
    }

    public override void Flush()
    {
        primary.Flush();
        secondary.Flush();
    }
}

internal sealed class Handshake
{
    [JsonPropertyName("protocol_version")] public int ProtocolVersion { get; init; }
    [JsonPropertyName("kind")] public string Kind { get; init; } = "";
    [JsonPropertyName("run_token")] public string RunToken { get; init; } = "";
    [JsonPropertyName("process_id")] public int ProcessId { get; init; }
    [JsonPropertyName("executable_sha256")] public string ExecutableSha256 { get; init; } = "";
    [JsonPropertyName("beatmap_sha256")] public string BeatmapSha256 { get; init; } = "";
    [JsonPropertyName("beatmap_md5")] public string BeatmapMd5 { get; init; } = "";
    [JsonPropertyName("mods")] public string[] Mods { get; init; } = Array.Empty<string>();
    [JsonPropertyName("clock_rate")] public double ClockRate { get; init; }
    [JsonPropertyName("gameplay_start_time_ms")] public double GameplayStartTimeMs { get; init; }
    [JsonPropertyName("qpc_frequency")] public long QpcFrequency { get; init; }
    [JsonPropertyName("playfield_origin")] public double[] PlayfieldOrigin { get; init; } = Array.Empty<double>();
    [JsonPropertyName("playfield_x_axis")] public double[] PlayfieldXAxis { get; init; } = Array.Empty<double>();
    [JsonPropertyName("playfield_y_axis")] public double[] PlayfieldYAxis { get; init; } = Array.Empty<double>();
}

internal sealed class SelectionMessage
{
    [JsonPropertyName("protocol_version")] public int ProtocolVersion { get; init; }
    [JsonPropertyName("kind")] public string Kind { get; init; } = "";
    [JsonPropertyName("run_token")] public string RunToken { get; init; } = "";
    [JsonPropertyName("beatmap_sha256")] public string BeatmapSha256 { get; init; } = "";
    [JsonPropertyName("beatmap_md5")] public string BeatmapMd5 { get; init; } = "";
    [JsonPropertyName("mods")] public string[] Mods { get; init; } = Array.Empty<string>();
    [JsonPropertyName("clock_rate")] public double ClockRate { get; init; }
}

internal sealed class StartMessage
{
    [JsonPropertyName("protocol_version")] public int ProtocolVersion { get; init; }
    [JsonPropertyName("kind")] public string Kind { get; init; } = "";
    [JsonPropertyName("run_token")] public string RunToken { get; init; } = "";
    [JsonPropertyName("qpc")] public long Qpc { get; init; }
    [JsonPropertyName("gameplay_clock_time_ms")] public double GameplayClockTimeMs { get; init; }
    [JsonPropertyName("playfield_origin")] public double[] PlayfieldOrigin { get; init; } = Array.Empty<double>();
    [JsonPropertyName("playfield_x_axis")] public double[] PlayfieldXAxis { get; init; } = Array.Empty<double>();
    [JsonPropertyName("playfield_y_axis")] public double[] PlayfieldYAxis { get; init; } = Array.Empty<double>();
}

internal sealed record Acknowledgement(
    [property: JsonPropertyName("accepted")] bool Accepted,
    [property: JsonPropertyName("run_token")] string RunToken,
    [property: JsonPropertyName("reason")] string? Reason);

internal readonly record struct WindowGuard(Native.RECT Rect, Native.RECT ClientRect, Native.RECT VirtualScreen, uint Dpi, int ProcessId)
{
    public static WindowGuard Capture(nint window, int processId)
    {
        if (OperatingSystem.IsWindows())
        {
            if (!Native.GetWindowRect(window, out var rect)) throw new InvalidOperationException("Cannot read research client window bounds.");
            Native.RECT clientRect = Native.GetClientScreenRect(window);
            Native.RECT virtualScreen = Native.GetVirtualScreenRect();
            return new WindowGuard(rect, clientRect, virtualScreen, Native.GetDpiForWindow(window), processId);
        }

        throw new PlatformNotSupportedException("Global window guards apply only to Windows.");
    }

    public void Validate(nint window)
    {
        if (OperatingSystem.IsWindows())
        {
            Native.GetWindowThreadProcessId(window, out uint pid);
            if (pid != ProcessId) throw new InvalidOperationException("Research window ownership changed; run aborted.");
            if (!Native.GetWindowRect(window, out var rect) || rect != Rect) throw new InvalidOperationException("Research window moved or resized; run aborted.");
            if (Native.GetClientScreenRect(window) != ClientRect) throw new InvalidOperationException("Research client size or screen position changed; run aborted.");
            if (Native.GetVirtualScreenRect() != VirtualScreen) throw new InvalidOperationException("Virtual screen size or layout changed; run aborted.");
            if (Native.GetDpiForWindow(window) != Dpi) throw new InvalidOperationException("Research window DPI changed; run aborted.");
        }
        else
        {
            throw new PlatformNotSupportedException("Global window guards apply only to Windows.");
        }
    }

    public void ValidatePlayfield(StartMessage start)
        => ValidatePlayfield(start.PlayfieldOrigin, start.PlayfieldXAxis, start.PlayfieldYAxis);

    public void ValidatePlayfield(double[] originValues, double[] xAxisValues, double[] yAxisValues)
    {
        if (originValues.Length != 2 || xAxisValues.Length != 2 || yAxisValues.Length != 2)
            throw new InvalidDataException("Physical playfield transform is invalid.");
        double scale = 96.0 / Dpi;
        (double X, double Y) origin = (originValues[0] * scale, originValues[1] * scale);
        (double X, double Y) xAxis = (xAxisValues[0] * scale, xAxisValues[1] * scale);
        (double X, double Y) yAxis = (yAxisValues[0] * scale, yAxisValues[1] * scale);
        (double X, double Y) opposite = (xAxis.X + yAxis.X - origin.X, xAxis.Y + yAxis.Y - origin.Y);
        double xDx = xAxis.X - origin.X, xDy = xAxis.Y - origin.Y;
        double yDx = yAxis.X - origin.X, yDy = yAxis.Y - origin.Y;
        double xLength = Math.Sqrt(xDx * xDx + xDy * xDy);
        double yLength = Math.Sqrt(yDx * yDx + yDy * yDy);
        if (xLength < 1 || yLength < 1) throw new InvalidDataException("Physical playfield transform is degenerate.");
        double aspect = xLength / yLength;
        if (Math.Abs(aspect - 4.0 / 3.0) > 0.02)
            throw new InvalidDataException($"Physical playfield ratio is invalid ({aspect:F4}:1, expected 1.3333:1).");
        double orthogonality = Math.Abs(xDx * yDx + xDy * yDy) / (xLength * yLength);
        if (orthogonality > 0.01)
            throw new InvalidDataException($"Physical playfield axes are not orthogonal ({orthogonality:F4}).");
        foreach ((double X, double Y) point in new[] { origin, xAxis, yAxis, opposite })
        {
            if (point.X < ClientRect.Left - 2 || point.X > ClientRect.Right + 2 || point.Y < ClientRect.Top - 2 || point.Y > ClientRect.Bottom + 2)
                throw new InvalidDataException($"Physical playfield point ({point.X:F2},{point.Y:F2}) is outside the {ClientRect.Width}x{ClientRect.Height} research client.");
        }
        Console.WriteLine($"Physical playfield: O=({origin.X:F2},{origin.Y:F2}) X=({xAxis.X:F2},{xAxis.Y:F2}) Y=({yAxis.X:F2},{yAxis.Y:F2}); {xLength:F2}x{yLength:F2}, ratio {aspect:F4}:1.");
    }
}

internal static class Input
{
    private static readonly HashSet<ushort> heldKeys = new();
    private static readonly Native.INPUT[] frameEvents = new Native.INPUT[3];
    public static string BackendId => OperatingSystem.IsWindows() ? "windows-sendinput" : "timing-only";
    public static string BackendName => OperatingSystem.IsWindows() ? "SendInput" : "timing-only";

    public static void SendFrame(double x, double y, bool targetK1, bool targetK2, ref bool k1, ref bool k2, ushort leftKey, ushort rightKey)
    {
        if (!OperatingSystem.IsWindows())
        {
            throw new PlatformNotSupportedException("Linux input must remain confined to the research client.");
        }

        // SendInput inserts this array serially and without interleaving. Keeping
        // the mouse move first preserves cursor-before-key ordering while using
        // one kernel transition instead of two on every hit frame. Absolute
        // moves teleport any distance in a single event.
        int eventCount = 0;
        frameEvents[eventCount++] = Native.MouseMove(x, y);
        if (targetK1 != k1)
        {
            if (targetK1) heldKeys.Add(leftKey);
            frameEvents[eventCount++] = Native.Key(leftKey, !targetK1);
        }
        if (targetK2 != k2)
        {
            if (targetK2) heldKeys.Add(rightKey);
            frameEvents[eventCount++] = Native.Key(rightKey, !targetK2);
        }

        send(frameEvents, eventCount);
        if (targetK1 != k1)
        {
            track(leftKey, targetK1);
            k1 = targetK1;
        }
        if (targetK2 != k2)
        {
            track(rightKey, targetK2);
            k2 = targetK2;
        }
    }

    public static void ReleaseKey(ushort virtualKey)
    {
        if (!heldKeys.Remove(virtualKey)) return;
        if (OperatingSystem.IsWindows()) send(new[] { Native.Key(virtualKey, true) }, 1);
        else throw new PlatformNotSupportedException("No global Linux key backend exists.");
    }

    public static void ReleaseAllKeys()
    {
        foreach (ushort key in heldKeys.ToArray())
        {
            try
            {
                if (OperatingSystem.IsWindows()) send(new[] { Native.Key(key, true) }, 1);
            }
            catch { }
            heldKeys.Remove(key);
        }
    }

    private static void track(ushort key, bool down)
    {
        if (down) heldKeys.Add(key);
        else heldKeys.Remove(key);
    }

    [SupportedOSPlatform("windows")]
    private static void send(Native.INPUT[] values, int count)
    {
        uint sent = Native.SendInput((uint)count, values, Marshal.SizeOf<Native.INPUT>());
        if (sent != count) throw new InvalidOperationException($"SendInput wrote {sent} of {count} events; run aborted.");
    }
}

internal static class Native
{
    private const int SM_XVIRTUALSCREEN = 76;
    private const int SM_YVIRTUALSCREEN = 77;
    private const int SM_CXVIRTUALSCREEN = 78;
    private const int SM_CYVIRTUALSCREEN = 79;
    private const uint INPUT_MOUSE = 0;
    private const uint INPUT_KEYBOARD = 1;
    private const uint MOUSEEVENTF_MOVE = 0x0001;
    private const uint MOUSEEVENTF_VIRTUALDESK = 0x4000;
    private const uint MOUSEEVENTF_ABSOLUTE = 0x8000;
    private const uint KEYEVENTF_KEYUP = 0x0002;
    private const uint KEYEVENTF_SCANCODE = 0x0008;
    private const uint MAPVK_VK_TO_VSC = 0;

    [SupportedOSPlatform("windows")]
    [DllImport("user32.dll", SetLastError = true)] internal static extern uint SendInput(uint count, INPUT[] inputs, int size);
    [SupportedOSPlatform("windows")]
    [DllImport("user32.dll")] internal static extern nint GetForegroundWindow();
    [SupportedOSPlatform("windows")]
    [DllImport("user32.dll")] internal static extern uint GetWindowThreadProcessId(nint window, out uint processId);
    [SupportedOSPlatform("windows")]
    [DllImport("user32.dll", SetLastError = true)] [return: MarshalAs(UnmanagedType.Bool)] internal static extern bool GetWindowRect(nint window, out RECT rect);
    [SupportedOSPlatform("windows")]
    [DllImport("user32.dll", SetLastError = true)] [return: MarshalAs(UnmanagedType.Bool)] private static extern bool GetClientRect(nint window, out RECT rect);
    [SupportedOSPlatform("windows")]
    [DllImport("user32.dll", SetLastError = true)] [return: MarshalAs(UnmanagedType.Bool)] private static extern bool ClientToScreen(nint window, ref POINT point);
    [SupportedOSPlatform("windows")]
    [DllImport("user32.dll")] internal static extern uint GetDpiForWindow(nint window);
    [SupportedOSPlatform("windows")]
    [DllImport("winmm.dll", SetLastError = true)] internal static extern uint timeBeginPeriod(uint uPeriod);
    [SupportedOSPlatform("windows")]
    [DllImport("winmm.dll", SetLastError = true)] internal static extern uint timeEndPeriod(uint uPeriod);
    [SupportedOSPlatform("windows")]
    [DllImport("user32.dll")] private static extern int GetSystemMetrics(int index);
    [SupportedOSPlatform("windows")]
    [DllImport("user32.dll")] private static extern uint MapVirtualKey(uint code, uint mapType);

    [SupportedOSPlatform("windows")]
    internal static RECT GetClientScreenRect(nint window)
    {
        if (!GetClientRect(window, out RECT client)) throw new InvalidOperationException("Cannot read research client bounds.");
        var origin = new POINT(0, 0);
        if (!ClientToScreen(window, ref origin)) throw new InvalidOperationException("Cannot locate research client on screen.");
        return new RECT(origin.X, origin.Y, origin.X + client.Width, origin.Y + client.Height);
    }

    [SupportedOSPlatform("windows")]
    internal static RECT GetVirtualScreenRect()
    {
        int x = GetSystemMetrics(SM_XVIRTUALSCREEN), y = GetSystemMetrics(SM_YVIRTUALSCREEN);
        return new RECT(x, y, x + GetSystemMetrics(SM_CXVIRTUALSCREEN), y + GetSystemMetrics(SM_CYVIRTUALSCREEN));
    }

    [SupportedOSPlatform("windows")]
    internal static INPUT MouseMove(double screenX, double screenY)
    {
        int vx = GetSystemMetrics(SM_XVIRTUALSCREEN), vy = GetSystemMetrics(SM_YVIRTUALSCREEN);
        int vw = GetSystemMetrics(SM_CXVIRTUALSCREEN), vh = GetSystemMetrics(SM_CYVIRTUALSCREEN);
        int dx = (int)Math.Round((screenX - vx) * 65535.0 / Math.Max(1, vw - 1));
        int dy = (int)Math.Round((screenY - vy) * 65535.0 / Math.Max(1, vh - 1));
        return new INPUT
        {
            type = INPUT_MOUSE,
            data = new InputUnion
            {
                mouse = new MOUSEINPUT { dx = dx, dy = dy, flags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK },
            },
        };
    }

    [SupportedOSPlatform("windows")]
    internal static INPUT Key(ushort virtualKey, bool up) => new()
    {
        type = INPUT_KEYBOARD,
        data = new InputUnion
        {
            keyboard = new KEYBDINPUT
            {
                scan = (ushort)MapVirtualKey(virtualKey, MAPVK_VK_TO_VSC),
                flags = KEYEVENTF_SCANCODE | (up ? KEYEVENTF_KEYUP : 0),
            },
        },
    };

    [StructLayout(LayoutKind.Sequential)] internal struct INPUT { public uint type; public InputUnion data; }
    [StructLayout(LayoutKind.Explicit)] internal struct InputUnion { [FieldOffset(0)] public MOUSEINPUT mouse; [FieldOffset(0)] public KEYBDINPUT keyboard; }
    [StructLayout(LayoutKind.Sequential)] internal struct MOUSEINPUT { public int dx; public int dy; public uint mouseData; public uint flags; public uint time; public nint extraInfo; }
    [StructLayout(LayoutKind.Sequential)] internal struct KEYBDINPUT { public ushort virtualKey; public ushort scan; public uint flags; public uint time; public nint extraInfo; }
    [StructLayout(LayoutKind.Sequential)] internal record struct POINT(int X, int Y);
    [StructLayout(LayoutKind.Sequential)] internal record struct RECT(int Left, int Top, int Right, int Bottom)
    {
        public int Width => Right - Left;
        public int Height => Bottom - Top;
        public double AspectRatio => Width / (double)Math.Max(1, Height);
    }
}
