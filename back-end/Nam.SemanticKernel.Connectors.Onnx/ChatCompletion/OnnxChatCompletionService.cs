using System.Text;
using Microsoft.ML.OnnxRuntimeGenAI;
using Nam.SemanticKernel.Connectors.Onnx.Settings;

namespace Nam.SemanticKernel.Connectors.Onnx.ChatCompletion;

public sealed class OnnxChatCompletionService
    : OnnxServiceBase<OnnxChatCompletionService>,
        IChatCompletionService, IDisposable
{
    private readonly ILogger<OnnxChatCompletionService> _logger;
    private Model? _model;
    private Tokenizer? _tokenizer;

    public OnnxChatCompletionService(string modelId, string modelPath,
        JsonSerializerOptions? jsonSerializerOptions = null, ILoggerFactory? loggerFactory = null) : base(modelId,
        modelPath, jsonSerializerOptions, loggerFactory)
    {
        _logger = loggerFactory?.CreateLogger<OnnxChatCompletionService>() ??
                  NullLogger<OnnxChatCompletionService>.Instance;
    }

    public async Task<IReadOnlyList<ChatMessageContent>> GetChatMessageContentsAsync(ChatHistory chatHistory,
        PromptExecutionSettings? executionSettings = null,
        Kernel? kernel = null, CancellationToken cancellationToken = default)
    {
        var lastUserMessage = chatHistory.LastOrDefault(x => x.Role == AuthorRole.User);
        var messageToSend = lastUserMessage?.Content ?? string.Empty;

        LogChatCompletionStarted(messageToSend);
        // Run inference using the ONNX model
        var responseMessages = new List<string>();

        await foreach (var content in GetChatCompletionAsync(chatHistory, executionSettings, cancellationToken))
        {
            responseMessages.Add(content);
        }

        // Add the assistant's message to the chat history
        var responseMessage = string.Join(" ", responseMessages);
        chatHistory.AddAssistantMessage(responseMessage);

        LogChatCompletionSucceeded(messageToSend);

        return new List<ChatMessageContent>
        {
            new(AuthorRole.Assistant, responseMessage),
        };
    }

    public async IAsyncEnumerable<StreamingChatMessageContent> GetStreamingChatMessageContentsAsync(ChatHistory chatHistory,
        PromptExecutionSettings? executionSettings = null, Kernel? kernel = null,
        CancellationToken cancellationToken = default)
    {
        var lastUserMessage = chatHistory.LastOrDefault(x => x.Role == AuthorRole.User);
        var messageToSend = lastUserMessage?.Content ?? string.Empty;

        LogChatCompletionStreamingStarted(messageToSend);

        var messages = new List<StreamingChatMessageContent>();

        var responseQueue = new Queue<bool>();


        await foreach (var content in this.GetChatCompletionAsync(chatHistory, executionSettings, cancellationToken).ConfigureAwait(false))
        {
            yield return new StreamingChatMessageContent(AuthorRole.Assistant, content, modelId: ModelId);
        }


        throw new NotImplementedException();
    }

    #region Semantic kernel methods

    private async IAsyncEnumerable<string> GetChatCompletionAsync(ChatHistory chatHistory,
        PromptExecutionSettings? executionSettings,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var onnxPromptExecutionSettings = GetOnnxPromptExecutionSettingsSettings(executionSettings);
        var prompt = GetPrompt(chatHistory);
        var tokens = GetTokenizer().Encode(prompt);

        using var generatorParams = new GeneratorParams(GetModel());
        UpdateGeneratorParamsFromPromptExecutionSettings(generatorParams, onnxPromptExecutionSettings);
        generatorParams.SetInputSequences(tokens);

        using var generator = new Generator(GetModel(), generatorParams);

        if (generator is null)
        {
            throw new InvalidOperationException("The generator cannot be null.");
        }

        var removeNextTokenStartingWithSpace = true;
        while (!generator.IsDone())
        {
            cancellationToken.ThrowIfCancellationRequested();

            yield return await Task.Run(() =>
            {
                generator.ComputeLogits();
                generator.GenerateNextToken();

                var outputTokens = generator.GetSequence(0);
                var newToken = outputTokens.Slice(outputTokens.Length - 1, 1);
                var output = GetTokenizer().Decode(newToken);

                if (removeNextTokenStartingWithSpace && output[0] == ' ')
                {
                    removeNextTokenStartingWithSpace = false;
                    output = output.TrimStart();
                }

                return output;
            }, cancellationToken).ConfigureAwait(false);
        }
    }

    private OnnxRuntimeGenAiPromptExecutionSettings GetOnnxPromptExecutionSettingsSettings(
        PromptExecutionSettings? executionSettings)
    {
        if (JsonSerializerOptionsInstance is not null)
            return OnnxRuntimeGenAiPromptExecutionSettings.FromExecutionSettings(executionSettings,
                JsonSerializerOptionsInstance);

        return OnnxRuntimeGenAiPromptExecutionSettings.FromExecutionSettings(executionSettings);
    }

    private string GetPrompt(ChatHistory chatHistory)
    {
        var promptBuilder = new StringBuilder();
        foreach (var message in chatHistory) promptBuilder.Append($"<|{message.Role}|>\n{message.Content}");
        promptBuilder.Append("<|end|>\n<|assistant|>");
        return promptBuilder.ToString();
    }

    private Model GetModel() => _model ??= new Model(ModelPath);

    private Tokenizer GetTokenizer() => _tokenizer ??= new Tokenizer(GetModel());

    private void UpdateGeneratorParamsFromPromptExecutionSettings(GeneratorParams generatorParams,
        OnnxRuntimeGenAiPromptExecutionSettings promptExecutionSetting)
    {
        // Set search options
        // we will use mainly top 7 keys
        if (promptExecutionSetting.TopP.HasValue)
            generatorParams.SetSearchOption("top_p", promptExecutionSetting.TopP.Value);
        if (promptExecutionSetting.TopK.HasValue)
            generatorParams.SetSearchOption("top_k", promptExecutionSetting.TopK.Value);
        if (promptExecutionSetting.Temperature.HasValue)
            generatorParams.SetSearchOption("temperature", promptExecutionSetting.Temperature.Value);
        if (promptExecutionSetting.RepetitionPenalty.HasValue)
            generatorParams.SetSearchOption("repetition_penalty", promptExecutionSetting.RepetitionPenalty.Value);
        if (promptExecutionSetting.PastPresentShareBuffer.HasValue)
            generatorParams.SetSearchOption("past_present_share_buffer",
                promptExecutionSetting.PastPresentShareBuffer.Value);
        if (promptExecutionSetting.NumReturnSequences.HasValue)
            generatorParams.SetSearchOption("num_return_sequences", promptExecutionSetting.NumReturnSequences.Value);
        if (promptExecutionSetting.NoRepeatNgramSize.HasValue)
            generatorParams.SetSearchOption("no_repeat_ngram_size", promptExecutionSetting.NoRepeatNgramSize.Value);
        if (promptExecutionSetting.MinTokens.HasValue)
            generatorParams.SetSearchOption("min_length", promptExecutionSetting.MinTokens.Value);
        if (promptExecutionSetting.MaxTokens.HasValue)
            generatorParams.SetSearchOption("max_length", promptExecutionSetting.MaxTokens.Value);
        if (promptExecutionSetting.LengthPenalty.HasValue)
            generatorParams.SetSearchOption("length_penalty", promptExecutionSetting.LengthPenalty.Value);
        if (promptExecutionSetting.EarlyStopping.HasValue)
            generatorParams.SetSearchOption("early_stopping", promptExecutionSetting.EarlyStopping.Value);
        if (promptExecutionSetting.DoSample.HasValue)
            generatorParams.SetSearchOption("do_sample", promptExecutionSetting.DoSample.Value);
        if (promptExecutionSetting.DiversityPenalty.HasValue)
            generatorParams.SetSearchOption("diversity_penalty", promptExecutionSetting.DiversityPenalty.Value);
    }

    #endregion

    #region logging methods

    private void LogChatCompletionStarted(string prompt,
        [CallerMemberName] string callerMethodName = "",
        [CallerLineNumber] int callerLineNumber = 0)
    {
        _logger.LogTrace("{callerMethodName}({callerLineNumber}) - Starting to generate chat completion for '{prompt}'",
            callerMethodName, callerLineNumber, prompt);
    }

    private void LogChatCompletionSucceeded(string prompt,
        [CallerMemberName] string callerMethodName = "",
        [CallerLineNumber] int callerLineNumber = 0)
    {
        _logger.LogTrace(
            "{callerMethodName}({callerLineNumber}) - Successfully generated chat completion for prompt '{prompt}'",
            callerMethodName, callerLineNumber, prompt);
    }

    private void LogChatCompletionStreamingStarted(string prompt,
        [CallerMemberName] string callerMethodName = "",
        [CallerLineNumber] int callerLineNumber = 0)
    {
        _logger.LogTrace(
            "{callerMethodName}({callerLineNumber}) - Starting to generate streaming chat completion for '{prompt}'",
            callerMethodName, callerLineNumber, prompt);
    }

    private void LogChatCompletionStreamingSucceeded(string prompt,
        [CallerMemberName] string callerMethodName = "",
        [CallerLineNumber] int callerLineNumber = 0)
    {
        _logger.LogTrace(
            "{callerMethodName}({callerLineNumber}) - Successfully generated streaming chat completion for prompt '{prompt}'",
            callerMethodName, callerLineNumber, prompt);
    }

    #endregion

    public void Dispose()
    {
        _tokenizer?.Dispose();
        _model?.Dispose();
    }
}