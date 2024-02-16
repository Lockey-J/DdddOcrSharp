using System.Text;
using System.Text.Json;

namespace DdddOcrSharp
{
    /// <summary>
    /// ddddocr拓展方法
    /// </summary>
    public class DdddOcrOptions
    {
        /// <summary>
        /// 导入模型对应字符
        /// </summary>
        public List<string> Charset { get; set; } = new List<string>();
        /// <summary>
        /// Word 属性
        /// </summary>
        public bool Word { get; set; } = false;
        /// <summary>
        /// 缩放尺寸
        /// </summary>
        public Size Resize { get; set; }
        /// <summary>
        /// 通道数
        /// </summary>
        public int Channel { get; set; } = 1;
        /// <summary>
        /// 转换成JSON的字符串
        /// </summary>
        /// <returns>返回文本数据</returns>
        public string ToJson()
        {
            JsonSerializerOptions jsonOptions = new();
            jsonOptions.Converters.Add(new SizeJsonConverter());
            return JsonSerializer.Serialize(this, jsonOptions);
        }
        /// <summary>
        /// 从json结构中读取数据
        /// </summary>
        /// <param name="json">json文本</param>
        /// <returns></returns>
        public static DdddOcrOptions? FromJson(string json)
        {
            JsonSerializerOptions jsonOptions = new();
            jsonOptions.Converters.Add(new SizeJsonConverter());
            return JsonSerializer.Deserialize<DdddOcrOptions>(json, jsonOptions);
        }
        /// <summary>
        /// 从json文件中读取数据
        /// </summary>
        /// <param name="path">json文件路径</param>
        /// <returns></returns>
        public static DdddOcrOptions? FromJsonFile(string path)
        {
            return FromJson(File.ReadAllText(path, Encoding.UTF8));
        }
    }
}
