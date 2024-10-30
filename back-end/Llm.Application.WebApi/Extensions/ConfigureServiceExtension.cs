using System.Diagnostics.CodeAnalysis;
using Llm.Application.WebApi.Contracts;
using Llm.Application.WebApi.Models;
using Llm.Application.WebApi.Services;
using Microsoft.SemanticKernel;

namespace Llm.Application.WebApi.Extensions;

public static class ConfigureServiceExtension
{
    public static void ConfigureLlmServices(this IServiceCollection services, IConfiguration configuration)
    {
        services.Configure<OnnxModelOptions>(configuration.GetSection("OnnxModel"));
        services.AddSingleton<ILlmService, LlmService>();
        services.AddLogging(configure => configure.AddConsole());
    }

    [Experimental("SKEXP0070")]
    public static async Task ConfigureSemanticKernel(this IServiceCollection services, IConfiguration configuration)
    {
        var modelPath = configuration.GetModelPath();
        var semanticKernel = Kernel.CreateBuilder()
            .AddOnnxRuntimeGenAIChatCompletion("phi-3", modelPath: modelPath)
            .Build();
    }

    public static void ConfigureCors(this IServiceCollection services)
    {
        services.AddCors(options =>
        {
            options.AddPolicy("CorsPolicy", builder =>
            {
                builder
                    .AllowAnyOrigin()
                    .AllowAnyMethod()
                    .AllowAnyHeader();
            });
        });
    }

    #region private variables

    private static string GetModelPath(this IConfiguration configuration)
    {
        return configuration["OnnxModel:ModelPath"]!;
    }

    #endregion
}