namespace Spectre.Console.Tests;

public static class VerifyConfiguration
{
    [ModuleInitializer]
    public static void Init()
    {
        Verifier.DerivePathInfo((sourceFile, projectDirectory, type, method) =>
        {
            var original = Expectations.Initialize(sourceFile, projectDirectory, type, method);
            var expectedRoot = System.Environment.GetEnvironmentVariable("REPOFORGE_EXPECTATION_ROOT")
                ?? throw new System.InvalidOperationException("Expected source root is required.");
            var diagnosticRoot = System.Environment.GetEnvironmentVariable("REPOFORGE_DIAGNOSTIC_ROOT")
                ?? throw new System.InvalidOperationException("Diagnostic output root is required.");
            var relative = System.IO.Path.GetRelativePath(expectedRoot, original.Directory);
            if (System.IO.Path.IsPathRooted(relative) || relative == ".."
                || relative.StartsWith(".." + System.IO.Path.DirectorySeparatorChar))
            {
                throw new System.InvalidOperationException("Expectation directory escapes the frozen source root.");
            }
            return new PathInfo(System.IO.Path.Combine(diagnosticRoot, relative), original.TypeName, original.MethodName);
        });
    }
}
