using Spectre.Console.Cli;
using Spectre.Console.Testing;
using Xunit;

namespace Spectre.Console.Cli.Tests;

// Public reproduction of the requested help alias; no reference patch or hidden labels used.
// 为什么要有它：通过正常CommandApp入口比较可见输出，而不是按某个源码修改位置验收。
public sealed class RepoForgePublicHelpAliasTests
{
    private static CommandAppTester CreateApp()
    {
        var app = new CommandAppTester();
        app.Configure(config =>
        {
            config.SetApplicationName("rf-help");
            config.AddCommand<SampleCommand>("run");
        });
        return app;
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void QuestionMarkAliasShowsTheSameHelp(bool commandLevel)
    {
        var app = CreateApp();
        var canonical = commandLevel ? app.Run("run", "--help") : app.Run("--help");
        var alias = commandLevel ? app.Run("run", "-?") : app.Run("-?");
        Assert.Equal(0, canonical.ExitCode);
        Assert.False(string.IsNullOrWhiteSpace(canonical.Output));
        Assert.Equal(canonical.ExitCode, alias.ExitCode);
        Assert.Equal(canonical.Output, alias.Output);
    }

    [Theory]
    [InlineData("-h", false)]
    [InlineData("--help", false)]
    [InlineData("-h", true)]
    [InlineData("--help", true)]
    public void ExistingHelpOptionsRemainAvailable(string option, bool commandLevel)
    {
        var app = CreateApp();
        var result = commandLevel ? app.Run("run", option) : app.Run(option);
        Assert.Equal(0, result.ExitCode);
        Assert.False(string.IsNullOrWhiteSpace(result.Output));
    }

    [Fact]
    public void NormalCommandStillExecutes()
    {
        Assert.Equal(17, CreateApp().Run("run").ExitCode);
    }

    [Fact]
    public void UnrelatedUnknownOptionStillFails()
    {
        Assert.NotEqual(0, CreateApp().Run("--repo-forge-never-known-flag").ExitCode);
    }

    public sealed class SampleCommand : Command
    {
        public override int Execute(CommandContext context) => 17;
    }
}
