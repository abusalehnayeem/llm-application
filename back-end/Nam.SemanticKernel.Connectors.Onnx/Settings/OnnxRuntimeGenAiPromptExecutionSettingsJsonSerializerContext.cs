namespace Nam.SemanticKernel.Connectors.Onnx.Settings;

[JsonSerializable(typeof(OnnxRuntimeGenAiPromptExecutionSettings))]
internal sealed partial class OnnxRuntimeGenAiPromptExecutionSettingsJsonSerializerContext : JsonSerializerContext
{
    public static readonly OnnxRuntimeGenAiPromptExecutionSettingsJsonSerializerContext ReadPermissive = new(
        new JsonSerializerOptions
        {
            AllowTrailingCommas = true,
            PropertyNameCaseInsensitive = true,
            ReadCommentHandling = JsonCommentHandling.Skip
        });
}