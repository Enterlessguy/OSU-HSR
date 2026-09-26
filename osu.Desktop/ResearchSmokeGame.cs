using System;
using osu.Game.Rulesets.Osu.Research;

namespace osu.Desktop
{
    /// <summary>Native client startup check in an isolated, disposable research storage directory.</summary>
    internal partial class ResearchSmokeGame : OsuGameDesktop
    {
        public bool Succeeded { get; private set; }

        protected override void LoadComplete()
        {
            base.LoadComplete();
            Scheduler.AddDelayed(() =>
            {
                try
                {
                    if (!Host.IsActive.Value)
                        throw new InvalidOperationException("Native research smoke window did not gain focus.");
                    Console.WriteLine(ResearchTraceInputHandler.VerifyContracts());
                    Console.WriteLine($"native_research_host=loaded;focus=true;session={Environment.GetEnvironmentVariable("XDG_SESSION_TYPE") ?? "unknown"};login=false;submission=false");
                    Succeeded = true;
                }
                catch (Exception error)
                {
                    Console.Error.WriteLine(error);
                }
                finally
                {
                    Host.Exit();
                }
            }, 3000);
        }
    }
}
