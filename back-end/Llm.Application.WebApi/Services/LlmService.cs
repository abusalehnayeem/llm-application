using Llm.Application.WebApi.Contracts;
using Llm.Application.WebApi.Models;
using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntimeGenAI;

namespace Llm.Application.WebApi.Services;

public class LlmService : ILlmService
{
    private readonly ILogger<LlmService> _logger;
    private readonly Model _model;
    private readonly OnnxModelOptions _options;
    private readonly Tokenizer _tokenizer;

    public LlmService(IOptions<OnnxModelOptions> options, ILogger<LlmService> logger)
    {
        _options = options.Value;
        _logger = logger;
        _logger.LogInformation("Loading ONNX model from {ModelPath}", _options.ModelPath);
        _model = new Model(_options.ModelPath);
        _tokenizer = new Tokenizer(_model);
    }

    public async Task<string> GenerateResponseAsync(string input)
    {
        try
        {
            _logger.LogInformation("Generating response for input: {Input}", input);
            var inputTokens = _tokenizer.Encode(input);
            var generatorParams = new GeneratorParams(_model);
            generatorParams.SetSearchOption("max_length", _options.MaxLength);
            generatorParams.SetInputSequences(inputTokens);

            var generator = new Generator(_model, generatorParams);
            var response = GenerateTokenByToken(generator);

            _logger.LogInformation("Generated response: {Response}", response);

            return await Task.FromResult(response);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating response");
            throw;
        }
    }

    private string GenerateTokenByToken(Generator? generator)
    {
        if (generator is null)
        {
            throw new ArgumentNullException(nameof(generator));
        }
        var output = string.Empty;

        while (!generator.IsDone())
        {
            generator.ComputeLogits();
            generator.GenerateNextToken();
            var outputTokens = generator.GetSequence(0);
            var newToken = outputTokens.Slice(outputTokens.Length - 1, 1);
            output += _tokenizer.Decode(newToken);
        }

        return output;
    }
}