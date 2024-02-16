using System.Text.Json;
using System.Text.Json.Serialization;

namespace DdddOcrSharp
{
    /// <summary>
    /// JsonConverter OpenCvSharp.Size
    /// </summary>
    internal class SizeJsonConverter : JsonConverter<Size>
    {
        public override Size Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
          
            if (reader.TokenType != JsonTokenType.StartArray)
            {
                throw new JsonException("Size数据错误:[width,height]");
            }

            reader.Read();
            if (reader.TokenType != JsonTokenType.Number)
            {
                throw new JsonException("Size数据错误:[width,height]");
            }
            int width = reader.GetInt32();

            reader.Read();
            if (reader.TokenType != JsonTokenType.Number)
            {
                throw new JsonException("Size数据错误:[width,height]");
            }
            int height = reader.GetInt32();

            reader.Read();
            if (reader.TokenType != JsonTokenType.EndArray)
            {
                throw new JsonException("Size数据错误:[width,height]");
            }

            return new Size(width, height);
        }

        public override void Write(Utf8JsonWriter writer, Size value, JsonSerializerOptions options)
        {
            writer.WriteStartArray();
            writer.WriteNumberValue(value.Width);
            writer.WriteNumberValue(value.Height);
            writer.WriteEndArray();
        }
    }
}
