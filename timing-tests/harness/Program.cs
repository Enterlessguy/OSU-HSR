using System.Diagnostics;
using System.IO.Compression;
using System.IO.Pipes;
using System.Security.Cryptography;
using System.Text.Json;

string? pipeName = Environment.GetEnvironmentVariable("HUMAN_SIM_PIPE");
string? token = Environment.GetEnvironmentVariable("HUMAN_SIM_RUN_TOKEN");
string tracePath = Environment.GetEnvironmentVariable("SMOKE_TRACE") ?? string.Empty;
int heartbeatMs = int.Parse(Environment.GetEnvironmentVariable("SMOKE_HEARTBEAT_MS") ?? "50");
double driftPpm = double.Parse(Environment.GetEnvironmentVariable("SMOKE_DRIFT_PPM") ?? "0");

if (pipeName == null || token == null || !File.Exists(tracePath))
{
    Console.Error.WriteLine("missing env or trace");
    return 2;
}

string? headerLine = null;
long lastTimeUs = 0;
using (FileStream file = File.OpenRead(tracePath))
using (GZipStream gzip = new GZipStream(file, CompressionMode.Decompress))
using (StreamReader streamReader = new StreamReader(gzip))
{
    bool first = true;
    string? line;
    while ((line = streamReader.ReadLine()) != null)
    {
        if (first)
        {
            headerLine = line;
            first = false;
            continue;
        }

        using JsonDocument frame = JsonDocument.Parse(line);
        lastTimeUs = Math.Max(lastTimeUs, frame.RootElement.GetProperty("time_us").GetInt64());
    }
}

using JsonDocument header = JsonDocument.Parse(headerLine ?? throw new InvalidDataException("empty trace"));
string beatmapSha = header.RootElement.GetProperty("beatmap_sha256").GetString() ?? string.Empty;
string beatmapMd5 = header.RootElement.GetProperty("beatmap_md5").GetString() ?? string.Empty;
double timelineStart = header.RootElement.GetProperty("timeline_start_effective_ms").GetDouble();
double clockRate = header.RootElement.GetProperty("clock_rate").GetDouble();
string[] mods = header.RootElement.GetProperty("mods").EnumerateArray().Select(e => e.GetString() ?? string.Empty).ToArray();

string executableSha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(Environment.ProcessPath!))).ToLowerInvariant();
var rng = new Random(42);

using var pipe = new NamedPipeClientStream(".", pipeName, PipeDirection.InOut, PipeOptions.Asynchronous);
pipe.Connect(5000);
using var reader = new StreamReader(pipe, leaveOpen: true);
using var writer = new StreamWriter(pipe, leaveOpen: true) { AutoFlush = true };

await writer.WriteLineAsync(JsonSerializer.Serialize(new
{
    protocol_version = 1,
    kind = "hello",
    run_token = token,
    process_id = Environment.ProcessId,
    executable_sha256 = executableSha256,
    beatmap_sha256 = beatmapSha,
    beatmap_md5 = beatmapMd5,
    mods = mods,
    clock_rate = clockRate,
    gameplay_start_time_ms = 0.0,
    qpc_frequency = Stopwatch.Frequency,
    playfield_origin = new double[] { 0, 0 },
    playfield_x_axis = new double[] { 512, 0 },
    playfield_y_axis = new double[] { 0, 384 },
}));

string? ack = await reader.ReadLineAsync();
Console.WriteLine("ACK: " + ack);
if (ack?.Contains("\"accepted\":true", StringComparison.OrdinalIgnoreCase) != true)
    return 3;

long startQpc = Stopwatch.GetTimestamp();
Console.WriteLine($"[wall] harness start {DateTimeOffset.UtcNow:HH:mm:ss.fff}");
await writer.WriteLineAsync(JsonSerializer.Serialize(new
{
    protocol_version = 1,
    kind = "start",
    run_token = token,
    qpc = startQpc,
    gameplay_clock_time_ms = timelineStart,
    playfield_origin = new double[] { 0, 0 },
    playfield_x_axis = new double[] { 512, 0 },
    playfield_y_axis = new double[] { 0, 384 },
}));

long endUs = lastTimeUs + 1_500_000;
long nextHeartbeatUs = 0;
while (true)
{
    long nowUs = (Stopwatch.GetTimestamp() - startQpc) * 1_000_000L / Stopwatch.Frequency;
    if (nowUs >= endUs)
        break;
    if (nowUs < nextHeartbeatUs)
    {
        await Task.Delay(1);
        continue;
    }

    double elapsedMs = (Stopwatch.GetTimestamp() - startQpc) * 1000.0 / Stopwatch.Frequency;
    double clockMs = timelineStart + elapsedMs * (1.0 + driftPpm / 1e6) + (rng.NextDouble() - 0.5) * 1.0;
    try
    {
        await writer.WriteLineAsync(JsonSerializer.Serialize(new
        {
            kind = "heartbeat",
            run_token = token,
            qpc = Stopwatch.GetTimestamp(),
            gameplay_clock_time_ms = clockMs,
        }));
    }
    catch (IOException)
    {
        break;
    }

    nextHeartbeatUs += heartbeatMs * 1000L;
}

Console.WriteLine("harness done");
Console.WriteLine($"[wall] harness heartbeat end {DateTimeOffset.UtcNow:HH:mm:ss.fff}");
// Keep the pipe open briefly so the runner can finish writing its completion
// message after the final trace frame.
await Task.Delay(3000);
return 0;
