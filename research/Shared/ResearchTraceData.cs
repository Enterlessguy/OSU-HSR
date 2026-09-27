using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;

namespace HumanSim.Transport
{
    /// <summary>Bounded, authenticated transfer of a synthetic timeline to the owned research client.</summary>
    public sealed class ResearchTraceData
    {
        public const int MaximumFrames = 1_800_000;
        public int Version { get; set; } = 1;
        public bool Synthetic { get; set; } = true;
        public string RunToken { get; set; } = "";
        public string BeatmapSha256 { get; set; } = "";
        public double ClockRate { get; set; }
        public double TimelineStartEffectiveMs { get; set; }
        public string ExecutionMode { get; set; } = "math-only";
        public int LearnedSegments { get; set; }
        public int ChangedSamples { get; set; }
        public int FallbackSegments { get; set; }
        public List<ResearchTraceFrame> Frames { get; set; } = new();

        public void Validate(string token, string beatmapHash, double rate)
        {
            if (Version != 1 || !Synthetic || RunToken != token || BeatmapSha256 != beatmapHash)
                throw new InvalidDataException("Research trace identity mismatch.");
            if (!double.IsFinite(ClockRate) || ClockRate <= 0 || Math.Abs(ClockRate - rate) > 0.000001
                || !double.IsFinite(TimelineStartEffectiveMs))
                throw new InvalidDataException("Research trace clock mismatch.");
            if (ExecutionMode is not ("coherent" or "math-only" or "hybrid"))
                throw new InvalidDataException("Unsupported research execution mode.");
            if (LearnedSegments < 0 || ChangedSamples < 0 || FallbackSegments < 0
                || (ExecutionMode == "coherent" && (LearnedSegments == 0 || ChangedSamples == 0)))
                throw new InvalidDataException("Research trace has no learned exposure or invalid counters.");
            if (Frames == null || Frames.Count == 0 || Frames.Count > MaximumFrames)
                throw new InvalidDataException("Research trace frame limit exceeded.");
            long previous = -1;
            foreach (ResearchTraceFrame frame in Frames)
            {
                if (frame == null || frame.TimeUs <= previous || frame.TimeUs > 1_800_000_000
                    || !double.IsFinite(frame.X) || !double.IsFinite(frame.Y)
                    || frame.X is < -512 or > 1024 || frame.Y is < -512 or > 896)
                    throw new InvalidDataException("Invalid research trace frame.");
                previous = frame.TimeUs;
            }
            if (Frames[^1].K1 || Frames[^1].K2)
                throw new InvalidDataException("Research trace ends with held input.");
        }

        public string Write(string path)
        {
            var options = new FileStreamOptions { Mode = FileMode.CreateNew, Access = FileAccess.Write, Share = FileShare.None };
            if (!OperatingSystem.IsWindows()) options.UnixCreateMode = UnixFileMode.UserRead | UnixFileMode.UserWrite;
            using (var file = new FileStream(path, options))
            {
                using var compressed = new GZipStream(file, CompressionLevel.Fastest);
                JsonSerializer.Serialize(compressed, this);
            }
            using var source = File.OpenRead(path);
            return Convert.ToHexString(SHA256.HashData(source)).ToLowerInvariant();
        }

        public static ResearchTraceData Read(string path, string expectedHash)
        {
            if (!Path.IsPathFullyQualified(path) || expectedHash.Length != 64)
                throw new InvalidDataException("Invalid research trace descriptor.");
            using var file = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
            if (file.Length > 64 * 1024 * 1024) throw new InvalidDataException("Research trace file too large.");
            string actualHash = Convert.ToHexString(SHA256.HashData(file)).ToLowerInvariant();
            if (!CryptographicOperations.FixedTimeEquals(Convert.FromHexString(actualHash), Convert.FromHexString(expectedHash)))
                throw new InvalidDataException("Research trace digest mismatch.");
            file.Position = 0;
            using var compressed = new GZipStream(file, CompressionMode.Decompress);
            using var bounded = new MemoryStream();
            byte[] buffer = new byte[81920];
            int count;
            while ((count = compressed.Read(buffer, 0, buffer.Length)) != 0)
            {
                if (bounded.Length + count > 256 * 1024 * 1024)
                    throw new InvalidDataException("Research trace decompression limit exceeded.");
                bounded.Write(buffer, 0, count);
            }
            bounded.Position = 0;
            return JsonSerializer.Deserialize<ResearchTraceData>(bounded) ?? throw new InvalidDataException("Empty research trace payload.");
        }

        public static int VerifyContracts()
        {
            int checks = 0;
            ResearchTraceData valid() => new()
            {
                RunToken = "contract-test", BeatmapSha256 = new string('b', 64), ClockRate = 1,
                Frames = new List<ResearchTraceFrame>
                {
                    new() { TimeUs = 0, X = 100, Y = 200, K1 = true },
                    new() { TimeUs = 1000, X = 101, Y = 200 },
                },
            };
            foreach (Action<ResearchTraceData> corrupt in new Action<ResearchTraceData>[]
            {
                trace => trace.RunToken = "wrong", trace => trace.BeatmapSha256 = "wrong",
                trace => trace.Synthetic = false, trace => trace.Version = 2,
                trace => trace.ClockRate = double.NaN, trace => trace.ClockRate = 1.5,
                trace => trace.Frames[1].TimeUs = 0, trace => trace.Frames[1].K2 = true,
                trace => trace.Frames[0].X = double.PositiveInfinity,
                trace => trace.Frames[0].Y = 10000, trace => trace.ExecutionMode = "coherent",
                trace => trace.Frames.Clear(),
            })
            {
                ResearchTraceData trace = valid();
                corrupt(trace);
                try { trace.Validate("contract-test", new string('b', 64), 1); }
                catch (InvalidDataException) { checks++; continue; }
                throw new InvalidDataException("An invalid research transport was accepted.");
            }
            string path = Path.Combine(Path.GetTempPath(), $"hsr-contract-{Guid.NewGuid():N}.json.gz");
            try
            {
                ResearchTraceData trace = valid();
                string hash = trace.Write(path);
                Read(path, hash).Validate("contract-test", new string('b', 64), 1);
                checks++;
                try { Read(path, new string('0', 64)); }
                catch (InvalidDataException) { checks++; return checks; }
                throw new InvalidDataException("A mismatched research transport digest was accepted.");
            }
            finally { File.Delete(path); }
        }
    }

    public sealed class ResearchTraceFrame
    {
        public long TimeUs { get; set; }
        public double X { get; set; }
        public double Y { get; set; }
        public bool K1 { get; set; }
        public bool K2 { get; set; }
    }
}
