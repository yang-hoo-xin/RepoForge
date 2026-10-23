using System.Collections.Generic;
using System.Linq;
using Serilog.Core;
using Serilog.Events;
using Xunit;

namespace Serilog.Tests;

// Public reproduction of the supplied issue; no implementation location or repair is prescribed.
public class ReportedIssue2103Tests
{
    sealed class Capture : ILogEventSink
    {
        public List<LogEvent> Events { get; } = new();
        public void Emit(LogEvent logEvent) => Events.Add(logEvent);
    }

    public class Inner
    {
        public string InnerObjectName { get; set; } = "InnerObject";
    }

    public class Outer
    {
        public string Name { get; set; } = "Test";
        public string Description { get; set; } = "My Description";
        public Inner InnerObject { private get; set; } = new();
    }

    [Fact]
    public void Reported_private_getter_is_not_in_the_destructured_output()
    {
        var capture = new Capture();
        using var logger = new LoggerConfiguration().WriteTo.Sink(capture).CreateLogger();
        logger.Information("Testing of destructuring! {@OuterObject}", new Outer());
        var value = Assert.IsType<StructureValue>(Assert.Single(capture.Events).Properties["OuterObject"]);
        Assert.DoesNotContain(value.Properties, p => p.Name == nameof(Outer.InnerObject));
        Assert.Equal("Test", Assert.IsType<ScalarValue>(value.Properties.Single(p => p.Name == nameof(Outer.Name)).Value).Value);
        Assert.Equal("My Description", Assert.IsType<ScalarValue>(value.Properties.Single(p => p.Name == nameof(Outer.Description)).Value).Value);
    }

    public class PublicReadOnly
    {
        public string Visible { get; } = "preserved";
    }

    [Fact]
    public void Ordinary_public_readable_property_remains_visible()
    {
        var capture = new Capture();
        using var logger = new LoggerConfiguration().WriteTo.Sink(capture).CreateLogger();
        logger.Information("Value {@Object}", new PublicReadOnly());
        var value = Assert.IsType<StructureValue>(Assert.Single(capture.Events).Properties["Object"]);
        Assert.Equal("preserved", Assert.IsType<ScalarValue>(value.Properties.Single(p => p.Name == nameof(PublicReadOnly.Visible)).Value).Value);
    }
}
