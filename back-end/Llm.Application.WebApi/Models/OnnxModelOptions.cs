namespace Llm.Application.WebApi.Models;

public class OnnxModelOptions
{
    public required string ModelPath { get; set; }
    public int MaxLength { get; set; } = 2048;
}