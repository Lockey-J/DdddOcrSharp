using System.ComponentModel;
using System.Reflection;
using System.Runtime.CompilerServices;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats;

namespace DdddOcrSharp
{
    /// <summary>
    /// DdddOcr拓展方法
    /// </summary>
    public static partial class DdddOcrExtensions
    {
        /// <summary>
        /// 获取属性的Description内容
        /// </summary>
        /// <param name="value">枚举值</param>
        /// <returns></returns>
        public static string GetDescription(this Enum value)
        {
            Type type = value.GetType();
            string? name = Enum.GetName(type, value) ?? string.Empty;
            FieldInfo? field = type.GetField(name);
            DescriptionAttribute? attribute = field?.GetCustomAttribute<DescriptionAttribute>();
            return attribute != null ? attribute.Description : name;
        }

        private static void EnsureMode(this DDDDOCR predictor, DdddOcrMode mode)
        {
            if (Enum.IsDefined(mode))
                throw new InvalidOperationException($"The mode does not support {mode}");
        }
        /// <summary>
        /// null情况下抛出错误
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="argument"></param>
        /// <param name="paramName"></param>
        /// <param name="methodName"></param>
        /// <exception cref="ArgumentNullException"></exception>
        public static void ThrowIfNull<T>(T argument, string paramName, [CallerMemberName] string methodName = "")
        {
            if (argument is null)
            {
                throw new ArgumentNullException(paramName, $"{methodName} => Parameter '{paramName}' cannot be null.");
            }
        }

        /// <summary>
        /// 判断图片类型
        /// </summary>
        /// <param name="imageBytes"></param>
        /// <returns></returns>
        public static IImageFormat GetImageFormat(this byte[] imageBytes)
        {
            return Image.DetectFormat(imageBytes);
        }

        /// <summary>
        /// GifToPng
        /// </summary>
        /// <param name="gifBytes"></param>
        /// <returns>返回png图片</returns>
        public static byte[] GifToPng(this byte[] gifBytes)
        {
            using var image = Image.Load(gifBytes);
            using var ms = new MemoryStream();
            image.SaveAsPng(ms);
            return ms.ToArray();
        }
    }
}
