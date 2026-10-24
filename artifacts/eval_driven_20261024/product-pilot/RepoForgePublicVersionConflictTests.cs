using System.ComponentModel;
using System.Text.RegularExpressions;
using Spectre.Console.Cli;
using Spectre.Console.Testing;
using Xunit;

namespace Spectre.Console.Cli.Tests;

// Public reproduction from the user's described custom option, not an implementation selector.
// 为什么要有它：检查实际帮助和执行行为，同时保留没有冲突时的内置版本功能。
public sealed class RepoForgePublicVersionConflictTests
{
    private static CommandAppTester CreateApp(bool customVersion)
    {
        var app = new CommandAppTester();
        if (customVersion)
        {
            app.SetDefaultCommand<UserVersionCommand>();
        }
        else
        {
            app.SetDefaultCommand<PlainCommand>();
        }

        app.Configure(config =>
        {
            config.SetApplicationName("version-fixture");
            if (!customVersion)
            {
                config.SetApplicationVersion("7.4.2");
            }
        });
        return app;
    }

    [Theory]
    [InlineData("--help")]
    [InlineData("-h")]
    public void CustomVersionDoesNotShowAConflictingBuiltInOption(string helpOption)
    {
        var help = CreateApp(true).Run(helpOption);
        Assert.Equal(0, help.ExitCode);
        Assert.Contains("User-supplied version switch", help.Output);
        Assert.DoesNotContain("Prints version information", help.Output);
        Assert.Equal(1, Regex.Matches(help.Output, "--version").Count);
    }

    [Fact]
    public void BuiltInHelpStillListsVersionWhenNoUserOptionConflicts()
    {
        var help = CreateApp(false).Run("--help");
        Assert.Equal(0, help.ExitCode);
        Assert.Contains("Prints version information", help.Output);
        Assert.Equal(1, Regex.Matches(help.Output, "--version").Count);
    }

    [Fact]
    public void BuiltInVersionStillExecutesWhenNoUserOptionConflicts()
    {
        // The public API's built-in version case has no default command.
        // 为什么要有它：保留已有根入口版本行为，不把默认命令的参数路由误当成版本缺陷。
        var app = new CommandAppTester();
        app.Configure(config => config.SetApplicationVersion("7.4.2"));
        var version = app.Run("--version");
        Assert.Equal(0, version.ExitCode);
        Assert.Contains("7.4.2", version.Output);
    }

    [Theory]
    [InlineData("--version")]
    [InlineData("-v")]
    public void ExistingCustomVersionExecutionRemainsOwnedByTheCommand(string option)
    {
        Assert.Equal(23, CreateApp(true).Run(option).ExitCode);
    }

    [Fact]
    public void NormalCommandWithoutVersionStillExecutes()
    {
        Assert.Equal(17, CreateApp(true).Run().ExitCode);
    }

    [Fact]
    public void UnrelatedUnknownOptionStillFails()
    {
        Assert.NotEqual(0, CreateApp(true).Run("--repo-forge-never-known-flag").ExitCode);
    }

    public sealed class UserVersionSettings : CommandSettings
    {
        [CommandOption("--version|-v")]
        [Description("User-supplied version switch.")]
        public bool CustomVersion { get; set; }
    }

    public sealed class UserVersionCommand : Command<UserVersionSettings>
    {
        public override int Execute(CommandContext context, UserVersionSettings settings)
            => settings.CustomVersion ? 23 : 17;
    }

    public sealed class PlainCommand : Command
    {
        public override int Execute(CommandContext context) => 17;
    }
}
