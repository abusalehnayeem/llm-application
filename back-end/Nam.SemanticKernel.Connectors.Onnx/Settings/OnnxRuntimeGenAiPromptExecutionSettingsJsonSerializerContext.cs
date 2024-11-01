namespace Nam.SemanticKernel.Connectors.Onnx.Settings;

#pragma warning disable SKEXP0001
[JsonSerializable(typeof(OnnxRuntimeGenAiPromptExecutionSettings))]
public sealed partial class OnnxRuntimeGenAiPromptExecutionSettingsJsonSerializerContext : JsonSerializerContext
{
    public static readonly OnnxRuntimeGenAiPromptExecutionSettingsJsonSerializerContext ReadPermissive = new(
        new JsonSerializerOptions
        {
            AllowTrailingCommas = true,
            PropertyNameCaseInsensitive = true,
            ReadCommentHandling = JsonCommentHandling.Skip
        });
}
#pragma warning restore SKEXP0021