namespace Llm.Application.WebApi.Contracts;

public interface ILlmService
{
    Task<string> GenerateResponseAsync(string input);
}