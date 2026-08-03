using osu.Game.Research;
using osu.Game.Rulesets.Mods;

if (!ResearchBuild.Enabled)
    throw new InvalidOperationException("HumanSimResearchBuild was not propagated to osu.Game.");
if (ResearchBuild.AllowsLogin)
    throw new InvalidOperationException("Research build unexpectedly permits login.");
if (ResearchBuild.AllowsScoreSubmission(Array.Empty<Mod>()))
    throw new InvalidOperationException("Research build unexpectedly permits score submission.");

Console.WriteLine("Verified: research build disables login and score submission.");
