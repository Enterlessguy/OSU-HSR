// Copyright (c) ppy Pty Ltd <contact@ppy.sh>. Licensed under the MIT Licence.
// See the LICENCE file in the repository root for full licence text.

using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;
using osu.Game.Beatmaps;
using osu.Game.Rulesets.Mods;
using osu.Game.Rulesets.Objects;
using osu.Game.Rulesets.Osu;
using osu.Game.Rulesets.Osu.Mods;
using osu.Game.Rulesets.Osu.Objects;
using osu.Game.Rulesets.Osu.Scoring;
using osu.Game.Rulesets.Scoring;
using osu.Game.Utils;
using osuTK;

namespace HumanSim.MapExporter;

public static class Program
{
    private static readonly JsonSerializerOptions json_options = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        WriteIndented = false,
    };

    public static int Main(string[] args)
    {
        try
        {
            var options = Options.Parse(args);
            export(options);
            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine(exception.Message);
            return 1;
        }
    }

    private static void export(Options options)
    {
        string sourcePath = Path.GetFullPath(options.MapPath);
        if (!File.Exists(sourcePath))
            throw new FileNotFoundException("Beatmap file was not found.", sourcePath);

        Mod[] mods = parseMods(options.Mods);
        if (options.ClockRate is double requestedClockRate)
        {
            if (!double.IsFinite(requestedClockRate) || requestedClockRate <= 0)
                throw new ArgumentException("Clock rate must be finite and positive.");

            ModRateAdjust? rateMod = mods.OfType<ModRateAdjust>().SingleOrDefault();
            if (rateMod == null)
            {
                if (Math.Abs(requestedClockRate - 1) > 1e-9)
                    throw new ArgumentException("A non-1.0 clock rate requires DT or HT.");
            }
            else
            {
                rateMod.SpeedChange.Value = requestedClockRate;
            }
        }
        double clockRate = ModUtils.CalculateRateWithMods(mods);
        if (options.ClockRate is double expectedClockRate && Math.Abs(clockRate - expectedClockRate) > 1e-9)
            throw new ArgumentException($"Requested clock rate {expectedClockRate:R} is outside the selected rate mod's supported range.");
        var working = new FlatWorkingBeatmap(sourcePath);
        var ruleset = new OsuRuleset();
        var playable = (IBeatmap<OsuHitObject>)working.GetPlayableBeatmap(ruleset.RulesetInfo, mods);
        if (playable.HitObjects.Count == 0)
            throw new InvalidDataException("Beatmap contains no osu!standard hit objects.");
        var hitWindows = new OsuHitWindows();
        hitWindows.SetDifficulty(playable.Difficulty.OverallDifficulty);

        string sha256;
        string md5;
        using (FileStream stream = File.OpenRead(sourcePath))
        {
            sha256 = Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
            stream.Position = 0;
            md5 = Convert.ToHexString(MD5.HashData(stream)).ToLowerInvariant();
        }

        var plan = new
        {
            SchemaVersion = 1,
            BeatmapSha256 = sha256,
            BeatmapMd5 = md5,
            ClockRate = clockRate,
            Mods = mods.Select(m => m.Acronym).Order().ToArray(),
            Metadata = new
            {
                playable.Metadata.Title,
                playable.Metadata.Artist,
                playable.Metadata.Author.Username,
                DifficultyName = playable.BeatmapInfo.DifficultyName,
                playable.Difficulty.ApproachRate,
                playable.Difficulty.OverallDifficulty,
                playable.Difficulty.CircleSize,
                playable.Difficulty.DrainRate,
                ObjectCount = playable.HitObjects.Count,
            },
            TimingPoints = playable.ControlPointInfo.TimingPoints.Select(point => new
            {
                EffectiveTimeMs = point.Time / clockRate,
                EffectiveBeatLengthMs = point.BeatLength / clockRate,
                point.TimeSignature,
            }),
            Objects = playable.HitObjects.Select((hitObject, index) => exportObject(hitObject, index, clockRate, hitWindows)).ToArray(),
        };

        string destination = Path.GetFullPath(options.OutputPath);
        Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
        using FileStream file = File.Create(destination);
        using var gzip = new GZipStream(file, CompressionLevel.SmallestSize);
        using var writer = new StreamWriter(gzip);
        writer.WriteLine(JsonSerializer.Serialize(plan, json_options));
        Console.WriteLine($"Exported {playable.HitObjects.Count} objects to {destination}");
        Console.WriteLine($"Beatmap SHA-256: {sha256}");
    }

    private static object exportObject(OsuHitObject hitObject, int index, double clockRate, HitWindows hitWindows)
    {
        string kind = hitObject switch
        {
            HitCircle => "circle",
            Slider => "slider",
            Spinner => "spinner",
            _ => throw new InvalidDataException($"Unsupported hit object type: {hitObject.GetType().Name}"),
        };
        double endTime = hitObject.GetEndTime();
        Vector2 actualEndPosition = hitObject switch
        {
            Slider endSlider when endSlider.RepeatCount % 2 == 1 => endSlider.StackedPosition,
            _ => hitObject.StackedEndPosition,
        };
        object[] pathSamples = hitObject is Slider sampledSlider
            ? Enumerable.Range(0, 101)
                        .Select(sample => point(sampledSlider.StackedPositionAt(sample / 100.0)))
                        .Cast<object>()
                        .ToArray()
            : Array.Empty<object>();

        return new
        {
            Index = index,
            Kind = kind,
            StartTimeMs = hitObject.StartTime,
            EndTimeMs = endTime,
            EffectiveStartTimeMs = hitObject.StartTime / clockRate,
            EffectiveEndTimeMs = endTime / clockRate,
            Position = point(hitObject.StackedPosition),
            EndPosition = point(actualEndPosition),
            hitObject.Radius,
            hitObject.StackHeight,
            RepeatCount = hitObject is Slider repeatSlider ? repeatSlider.RepeatCount : 0,
            PathSamples = pathSamples,
            HitWindows = new
            {
                GreatMs = hitWindows.WindowFor(HitResult.Great) / clockRate,
                OkMs = hitWindows.WindowFor(HitResult.Ok) / clockRate,
                MehMs = hitWindows.WindowFor(HitResult.Meh) / clockRate,
                MissMs = hitWindows.WindowFor(HitResult.Miss) / clockRate,
            },
        };
    }

    private static object point(Vector2 value) => new { X = value.X, Y = value.Y };

    private static Mod[] parseMods(string value)
    {
        var result = new List<Mod>();
        foreach (string acronym in value.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(m => m.ToUpperInvariant()))
        {
            result.Add(acronym switch
            {
                "HD" => new OsuModHidden(),
                "HR" => new OsuModHardRock(),
                "DT" => new OsuModDoubleTime(),
                "HT" => new OsuModHalfTime(),
                "FL" => new OsuModFlashlight(),
                _ => throw new ArgumentException($"Unsupported mod '{acronym}'. Supported: HD, HR, DT, HT, FL."),
            });
        }
        if (result.Any(m => m is OsuModDoubleTime) && result.Any(m => m is OsuModHalfTime))
            throw new ArgumentException("DT and HT cannot be combined.");
        return result.ToArray();
    }

    private sealed record Options(string MapPath, string OutputPath, string Mods, double? ClockRate)
    {
        public static Options Parse(string[] args)
        {
            string? map = null;
            string? output = null;
            string mods = string.Empty;
            double? clockRate = null;
            for (int i = 0; i < args.Length; i++)
            {
                switch (args[i])
                {
                    case "--map":
                        map = requireValue(args, ref i);
                        break;
                    case "--output":
                        output = requireValue(args, ref i);
                        break;
                    case "--mods":
                        mods = requireValue(args, ref i);
                        break;
                    case "--clock-rate":
                        if (!double.TryParse(requireValue(args, ref i), System.Globalization.NumberStyles.Float,
                                System.Globalization.CultureInfo.InvariantCulture, out double parsedClockRate))
                            throw new ArgumentException("Clock rate must be a number.");
                        clockRate = parsedClockRate;
                        break;
                    default:
                        throw new ArgumentException($"Unknown argument: {args[i]}");
                }
            }
            if (map == null || output == null)
                throw new ArgumentException("Usage: HumanSim.MapExporter --map <file.osu> --output <plan.ndjson.gz> [--mods HD,HR,DT,HT] [--clock-rate <rate>]");
            return new Options(map, output, mods, clockRate);
        }

        private static string requireValue(string[] args, ref int index)
        {
            if (++index >= args.Length)
                throw new ArgumentException("Missing argument value.");
            return args[index];
        }
    }
}
