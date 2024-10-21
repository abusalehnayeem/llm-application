using Llm.Application.WebApi.Contracts;
using Microsoft.AspNetCore.Mvc;

namespace Llm.Application.WebApi.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class WeatherForecastController(ILlmService llmService) : ControllerBase
    {
        [HttpPost("generate")]
        public async Task<IActionResult> Generate([FromBody] string input)
        {
            var response = await llmService.GenerateResponseAsync(input);
            return Ok(response);
        }
    }
}
