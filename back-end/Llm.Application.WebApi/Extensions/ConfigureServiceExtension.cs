using Llm.Application.WebApi.Contracts;
using Llm.Application.WebApi.Models;
using Llm.Application.WebApi.Services;
using Microsoft.Extensions.Configuration;

namespace Llm.Application.WebApi.Extensions;

public static class ConfigureServiceExtension
{
    public static void ConfigureServices(this IServiceCollection services, IConfiguration configuration)
    {
        services.Configure<OnnxModelOptions>(configuration.GetSection("OnnxModel"));
        services.AddSingleton<ILlmService, LlmService>();
        services.AddLogging(configure => configure.AddConsole());
    }
}