using System.ComponentModel;

namespace DdddOcrSharp
{
    /// <summary>
    /// DdddOcrMode枚举
    /// </summary>
    public enum DdddOcrMode
    {
        /// <summary>
        /// common_old.onnx
        /// </summary>
        [Description("onnxs\\common_old.onnx")]
        ClassifyOld,
        /// <summary>
        /// common.onnx
        /// </summary>
        [Description("onnxs\\common.onnx")]
        ClassifyBeta,
        /// <summary>
        /// common_det.onnx
        /// </summary>
        [Description("onnxs\\common_det.onnx")]
        Detect,
        /// <summary>
        /// 自定义导入
        /// </summary>
        Import
    }
}
