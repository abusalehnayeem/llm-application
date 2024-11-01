namespace Nam.SemanticKernel.Connectors.Onnx;
#pragma warning disable SKEXP0001
/// <summary>
/// Base class for ONNX services, providing common attributes and logging functionality.
/// </summary>
/// <typeparam name="T">The type of the ONNX service, which must derive from this class.</typeparam>
public abstract class OnnxServiceBase<T> where T : OnnxServiceBase<T>
{
    // Dictionary to store AI service attributes, initialized with default capacity.
    private readonly Dictionary<string, object?> _aiServiceAttributes = new();

    // Optional JSON serializer options for serialization and deserialization.
    protected readonly JsonSerializerOptions? JsonSerializerOptionsInstance; // Change to protected

    // Unique identifier for the ONNX model.
    protected readonly string ModelId;

    // File path to the ONNX model.
    protected readonly string ModelPath;

    // Logger instance for logging events and errors.
    protected readonly ILogger<T> Logger;

    /// <summary>
    /// Initializes a new instance of the <see cref="OnnxServiceBase{T}"/> class.
    /// </summary>
    /// <param name="modelId">Unique identifier for the ONNX model.</param>
    /// <param name="modelPath">File path to the ONNX model.</param>
    /// <param name="jsonSerializerOptions">Optional JSON serializer options.</param>
    /// <param name="loggerFactory">Optional logger factory for creating a logger instance.</param>
    /// <exception cref="ArgumentNullException">Thrown if <paramref name="modelId"/> or <paramref name="modelPath"/> is null.</exception>
    protected OnnxServiceBase(
        string modelId,
        string modelPath,
        JsonSerializerOptions? jsonSerializerOptions = null,
        ILoggerFactory? loggerFactory = null)
    {
        // Validate model ID and path.
        ModelId = modelId ?? throw new ArgumentNullException(nameof(modelId));
        ModelPath = modelPath ?? throw new ArgumentNullException(nameof(modelPath));

        // Store JSON serializer options.
        JsonSerializerOptionsInstance = jsonSerializerOptions;

        // Create logger instance using the provided factory, or use a null logger if not provided.
        Logger = loggerFactory?.CreateLogger<T>() ?? NullLogger<T>.Instance;

        // Add model ID as an attribute.
        AddAttribute(AIServiceExtensions.ModelIdKey, ModelId);
    }

    /// <summary>
    /// Gets a read-only dictionary of AI service attributes.
    /// </summary>
    public IReadOnlyDictionary<string, object?> Attributes => _aiServiceAttributes;

    /// <summary>
    /// Adds an attribute to the AI service attributes dictionary.
    /// </summary>
    /// <param name="key">The attribute key.</param>
    /// <param name="value">The attribute value, which must not be null or whitespace.</param>
    private void AddAttribute(string key, string? value)
    {
        // Only add the attribute if the value is not null or whitespace.
        if (!string.IsNullOrWhiteSpace(value)) _aiServiceAttributes.Add(key, value);
    }
}

#pragma warning restore SKEXP0021