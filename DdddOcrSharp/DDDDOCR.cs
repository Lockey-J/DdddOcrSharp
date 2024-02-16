using Microsoft.ML.OnnxRuntime.Tensors;
using System.Text;

namespace DdddOcrSharp
{
    /// <summary>
    /// DDDDOCR类
    /// </summary>
    public class DDDDOCR : IDisposable
    {
        private DdddOcrMode Mode { get; }
        private DdddOcrOptions? Options { get; }
        private SessionOptions? SessionOptions { get; }
        private InferenceSession InferenceSession { get; }

        #region Basic

        /// <summary>
        /// DDDDOCR使用自带模型实例化
        /// </summary>
        /// <param name="mode">实例化模型</param>
        /// <param name="use_gpu">是否使用GPU</param>
        /// <param name="device_id">GPU的ID</param>
        /// <exception cref="NotSupportedException">不支持模式报错</exception>
        /// <exception cref="FileNotFoundException">模型文件路径不存在报错</exception>
        public DDDDOCR(DdddOcrMode mode, bool use_gpu = false, int device_id = 0)
        {
#if DEBUG
            Console.WriteLine($"欢迎使用ddddocr，本项目专注带动行业内卷");
            Console.WriteLine($"python版开发作者：https://github.com/sml2h3/ddddocr");
            Console.WriteLine($"C#/NET版移植作者：https://github.com/itbencn/DdddOcr.Net");
            Console.WriteLine($"本项目仅作为移植项目未经过大量测试 生产环境谨慎使用");
            Console.WriteLine($"请勿违反所在地区法律法规 合理合法使用本项目");
#endif
            if (!Enum.IsDefined(mode))
            {
                throw new NotSupportedException($"不支持的模式:{mode}");
            }
            Mode = mode;
            Options = new DdddOcrOptions();
            var onnx_path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, Mode.GetDescription());
            if (!File.Exists(onnx_path))
            {
                throw new FileNotFoundException($"{mode}模式对应的模型文件不存在:{onnx_path}");
            }
            Options.Charset = Mode switch
            {
                DdddOcrMode.ClassifyOld => Global.OCR_OLD_CHARSET,
                DdddOcrMode.ClassifyBeta => Global.OCR_BETA_CHARSET,
                _ => Array.Empty<string>().ToList(),
            };
            SessionOptions = new SessionOptions();
            if (use_gpu)
            {
                SessionOptions.AppendExecutionProvider_CUDA(device_id);
            }
            else
            {
                SessionOptions.AppendExecutionProvider_CPU();
            }
            InferenceSession = new InferenceSession(File.ReadAllBytes(onnx_path), SessionOptions);
        }
        /// <summary>
        /// 使用自定义模型导入初始化方式
        /// </summary>
        /// <param name="import_onnx_path">导入模型路径</param>
        /// <param name="charsets_path">模型对应字符集路径</param>
        /// <param name="use_gpu">是否使用GPU</param>
        /// <param name="device_id">显卡GPU的ID</param>
        /// <exception cref="FileNotFoundException"></exception>
        /// <exception cref="FileLoadException"></exception>
        public DDDDOCR(string import_onnx_path, string charsets_path, bool use_gpu = false, int device_id = 0)
        {
#if DEBUG
            Console.WriteLine($"欢迎使用ddddocr，本项目专注带动行业内卷");
            Console.WriteLine($"python版开发作者：https://github.com/sml2h3/ddddocr");
            Console.WriteLine($"C#/NET版移植作者：https://github.com/itbencn/DdddOcr.Net");
            Console.WriteLine($"请合理合法使用本项目 本项目未经过大量测试 生产环境谨慎使用");
#endif
            Mode = DdddOcrMode.Import;
            if (!File.Exists(import_onnx_path))
            {
                throw new FileNotFoundException($"文件不存在:{import_onnx_path}");
            }
            if (!File.Exists(charsets_path))
            {
                throw new FileNotFoundException($"文件不存在:{charsets_path}");
            }
            Options = DdddOcrOptions.FromJsonFile(charsets_path);
            if (Options == null)
            {
                throw new FileLoadException("数据格式错误");
            }
            SessionOptions = new SessionOptions();
            if (use_gpu)
            {
                SessionOptions.AppendExecutionProvider_CUDA(device_id);
            }
            else
            {
                SessionOptions.AppendExecutionProvider_CPU();
            }
            InferenceSession = new InferenceSession(File.ReadAllBytes(import_onnx_path), SessionOptions);
        }

        /// <summary>
        /// 实例回收
        /// </summary>
        ~DDDDOCR()
        {
            Dispose();
        }
        /// <summary>
        /// ddddocr解构函数
        /// </summary>
        public void Dispose()
        {
            SessionOptions?.Dispose();
            InferenceSession?.Dispose();
            GC.SuppressFinalize(this);
        }
        #endregion

        #region classification
        /// <summary>
        /// OCR识别函数
        /// </summary>
        /// <param name="bytes">待识别图片字节集</param>
        /// <param name="pngFix">是否修复为png图片</param>
        /// <returns>返回识别文本</returns>
        /// <exception cref="InvalidOperationException"></exception>
        public string Classify(byte[] bytes, bool pngFix = false)
        {
            if (Mode == DdddOcrMode.Detect)
            {
                throw new InvalidOperationException("当前识别类型为目标检测");
            }
            using var image = Mat.FromImageData(bytes, ImreadModes.AnyColor);
            if (image.Width == 0 && image.Height == 0)
            {
                throw new InvalidOperationException("载入图像数据损坏或图片类型错误");
            }
            var inputs = ClassifyPrepareProcessing(image, pngFix);
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = InferenceSession.Run(inputs);
            var predictions = outputs.First(x => x.Name == "output").Value as DenseTensor<long>;
            if (predictions == null)
            {
                return string.Empty;
            }
            var result = new StringBuilder();
            foreach (long prediction in predictions)
            {
                result.Append(Options?.Charset[(int)prediction]);
            }
            return result.ToString();
        }

        static readonly float[] mean = { 0.485f, 0.456f, 0.406f };
        static readonly float[] std = { 0.229f, 0.224f, 0.225f };
        private List<NamedOnnxValue> ClassifyPrepareProcessing(Mat image, bool pngFix = false)
        {
            #region resize
            Mat resizedImg;
            if (Options == null)
                return new();
            if (Mode == DdddOcrMode.Import)
            {

                if (Options.Resize.Width == -1)
                {
                    if (Options.Word)
                    {
                        resizedImg = image.Resize(new Size(Options.Resize.Height, Options.Resize.Height), interpolation: InterpolationFlags.Linear);
                    }
                    else
                    {
                        resizedImg = image.Resize(new Size(image.Width * Convert.ToDouble(Options.Resize.Height / (double)image.Height), Options.Resize.Height), interpolation: InterpolationFlags.Linear);
                    }
                }
                else
                {
                    resizedImg = image.Resize(new Size(Options.Resize.Width, Options.Resize.Height), interpolation: InterpolationFlags.Linear);
                }

                if (Options.Channel == 1)
                {
                    //BGR2GRAY? RGB2GRAY?
                    resizedImg = resizedImg.CvtColor(ColorConversionCodes.BGR2GRAY);
                }
                else
                {
                    if (pngFix)
                    {
                        resizedImg = PngRgbaToRgbWhiteBackground(resizedImg);
                    }
                }
            }
            else
            {
                //BGR2GRAY? RGB2GRAY?
                resizedImg = image.Resize(new Size(image.Width * Convert.ToDouble(64d / image.Height), 64d), interpolation: InterpolationFlags.Linear).CvtColor(ColorConversionCodes.BGR2GRAY);
            }
            #endregion

            #region tensor
            int channels = resizedImg.Channels();
            var tensor = new DenseTensor<float>(new int[] { 1, channels, resizedImg.Height, resizedImg.Width });
            for (int y = 0; y < resizedImg.Height; y++)
            {
                for (int x = 0; x < resizedImg.Width; x++)
                {
                    if (Mode == DdddOcrMode.Import)
                    {
                        if (Options?.Channel == 1 || channels == 1)
                        {
                            byte color = resizedImg.Get<byte>(y, x);
                            tensor[0, 0, y, x] = ((color / 255f) - 0.456f) / 0.224f;
                        }
                        else
                        {
                            Vec3b color = resizedImg.Get<Vec3b>(y, x);
                            for (int c = 0; c < channels; c++)
                            {
                                tensor[0, c, y, x] = ((color[c] / 255f) - mean[c]) / std[c];
                            }
                        }
                    }
                    else
                    {
                        byte color = resizedImg.Get<byte>(y, x);
                        tensor[0, 0, y, x] = ((color / 255f) - 0.5f) / 0.5f;
                    }
                }
            }
            resizedImg.Dispose();
            return new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input1", tensor) };
            #endregion
        }

        private Mat PngRgbaToRgbWhiteBackground(Mat src)
        {
            if (src.Channels() != 4)
            {
                return src;
            }
            var whiteBackground = new Mat(src.Size(), MatType.CV_8UC3, Scalar.White);
            var srcChannels = Cv2.Split(src);
            using Mat alphaChannel = srcChannels[3];
            using var rgb = new Mat();
            Cv2.Merge(new[] { srcChannels[0], srcChannels[1], srcChannels[2] }, rgb);
            rgb.CopyTo(whiteBackground, alphaChannel);
            foreach (var mat in srcChannels)
            {
                mat.Dispose();
            }
            return whiteBackground;
        }
        #endregion

        #region detection
        /// <summary>
        /// 目标识别
        /// </summary>
        /// <param name="bytes">图片字节集</param>
        /// <returns>返回识别方框列表</returns>
        /// <exception cref="InvalidOperationException">初始识别类型错误</exception>
        public List<Rect> Detect(byte[] bytes)
        {
            if (Mode != DdddOcrMode.Detect)
            {
                throw new InvalidOperationException("当前识别类型为文字识别");
            }
            using var image = Mat.FromImageData(bytes, ImreadModes.AnyColor);
            if (image.Width == 0 && image.Height == 0)
            {
                throw new InvalidOperationException("载入图像数据损坏或图片类型错误");
            }
            var inputSize = new Size(416, 416);
            var inputs = DetectPrepareProcessing(image, inputSize);
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = InferenceSession.Run(inputs);
            var predictions = outputs.First(x => x.Name == "output").Value as DenseTensor<float>;
            var bboxs = DetectHandleProcessing(predictions, image);
            return bboxs;
        }

        private List<NamedOnnxValue> DetectPrepareProcessing(Mat image, Size inputSize)
        {
            #region resize
            //swap ??= new List<int> { 2, 0, 1 };
            Mat paddedImg;

            if (image.Channels() == 3)
            {
                paddedImg = new Mat(inputSize, MatType.CV_8UC3, new Scalar(114, 114, 114));
            }
            else
            {
                paddedImg = new Mat(inputSize, MatType.CV_8UC1, new Scalar(114));
            }

            float ratio = Math.Min((float)inputSize.Height / image.Rows, (float)inputSize.Width / image.Cols);
            var resizedImg = new Mat();
            Cv2.Resize(image, resizedImg, new Size((int)(image.Cols * ratio), (int)(image.Rows * ratio)), 0, 0, InterpolationFlags.Linear);

            resizedImg.CopyTo(paddedImg[new Rect(0, 0, resizedImg.Cols, resizedImg.Rows)]);


            #endregion

            #region tensor
            int channels = resizedImg.Channels();

            var tensor = new DenseTensor<float>(new int[] { 1, channels, inputSize.Height, inputSize.Width });
            for (int i = 0; i < paddedImg.Height; i++)
            {
                for (int j = paddedImg.Height - 1; j >= 0; j--)
                {
                    Vec3b color = paddedImg.Get<Vec3b>(i, j);
                    for (int c = 0; c < channels; c++)
                    {
                        tensor[0, c, i, j] = color[c];
                    }
                }
            }

            return new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", tensor) };
            #endregion
        }


        /// <summary>
        /// 目标识别的多类别NMSBoxs并返回识别后的目标Rect坐标
        /// </summary>
        /// <param name="output">DenseTensor识别数据</param>
        /// <param name="image">识别图片</param>
        /// <returns></returns>
        private List<Rect> DetectHandleProcessing(DenseTensor<float>? output, Mat image)
        {
            return DetectHandleProcessing(output, image, new Size(416, 416));
        }

        /// <summary>
        /// 目标识别的多类别NMSBoxs并返回识别后的目标Rect坐标
        /// </summary>
        /// <param name="output">DenseTensor识别数据</param>
        /// <param name="image">识别图片</param>
        /// <param name="img_size">默认图片大小</param>
        /// <param name="p6"></param>
        /// <param name="nms_thr">nms参数</param>
        /// <param name="score_thr">score参数</param>
        /// <returns></returns>
        private List<Rect> DetectHandleProcessing(DenseTensor<float>? output, Mat image, Size img_size, bool p6 = false, float nms_thr = 0.45f, float score_thr = 0.1f)
        {
            if (output == null)
                return new();

            List<float> scoreslist = new();
            List<Rect> bboxs = new();
            List<Rect> result = new();
            List<int> grids = new();
            List<int> expanded_strides = new();
            int[] strides;
            if (!p6)
            {
                strides = new int[] { 8, 16, 32 };
            }
            else
            {
                strides = new int[] { 8, 16, 32, 64 };
            }

            int[] hsizes = strides.Select(p => (int)(img_size.Height / p)).ToArray();
            int[] wsizes = strides.Select(p => (int)(img_size.Width / p)).ToArray();

            for (int i = 0; i < strides.Length; i++)
            {
                int hsize = hsizes[i], wsize = wsizes[i], stride = strides[i];
                var grid = MakeGrid(hsize, wsize);
                var expanded_stride = Makeexpanded_stride(hsize * wsize, stride);
                grids.AddRange(grid);
                expanded_strides.AddRange(expanded_stride);
            }
            var (w, h) = (image.Width, image.Height); // image w and h
            var (xGain, yGain) = (img_size.Width / (float)w, img_size.Height / (float)h); // x, y gains
            var gain = Math.Min(xGain, yGain); // gain = resized / original

            for (int i = 0; i < output.Length / 6; i++)
            {
                float scores = output[0, i, 4] * output[0, i, 5];

                float x1 = output[0, i, 0];

                float y1 = output[0, i, 1];

                float x2 = output[0, i, 2];

                float y2 = output[0, i, 3];

                x1 = (x1 + grids[i * 2 + 0]) * expanded_strides[i];
                y1 = (y1 + grids[i * 2 + 1]) * expanded_strides[i];
                x2 = (float)(Math.Exp(x2) * expanded_strides[i]);
                y2 = (float)(Math.Exp(y2) * expanded_strides[i]);

                float x11 = (x1 - x2 / 2) / gain;
                float y11 = (y1 - y2 / 2) / gain;
                float x22 = (x1 + x2 / 2) / gain;
                float y22 = (y1 + y2 / 2) / gain;

                scoreslist.Add(scores);

                bboxs.Add(new Rect((int)x11, (int)y11, (int)(x22 - x11), (int)(y22 - y11)));

            }

            CvDnn.NMSBoxes(bboxs, scoreslist, score_thr, nms_thr, out var indices);
            for (int i = 0; i < indices.Length; i++)
            {
                int index = indices[i];
                result.Add(bboxs[index]);

            }
            return result;
        }
        private int[] MakeGrid(int a, int b)
        {
            int[] ret = new int[a * b * 2];

            for (int i = 0; i < a; i++)
            {
                for (int j = 0; j < b; j += 1)
                {
                    int index = (i * a + j) * 2;
                    ret[index] = j;
                    ret[index + 1] = i;
                }
            }
            return ret;
        }

        private int[] Makeexpanded_stride(int a, int b)
        {
            int[] ret = new int[a];

            for (int i = 0; i < a; i++)
            {
                ret[i] = b;
            }
            return ret;
        }
        #endregion

        #region slide
        /// <summary>
        /// 滑块缺口识别算法1
        /// </summary>
        /// <param name="target">缺口背景图片</param>
        /// <param name="background">背景图片</param>
        /// <returns>返回缺口左上角坐标</returns>
        public static Point Slide_Comparison(Mat target, Mat background)
        {
            int start_x = 0, start_y = 0;
            // 将图像转换为灰度图，以便进行差异计算

            background = background.CvtColor(ColorConversionCodes.BGR2GRAY);
            target = target.CvtColor(ColorConversionCodes.BGR2GRAY);

            // 计算灰度图像的差异  
            Mat difference = new();
            Cv2.Absdiff(background, target, difference); // 使用OpenCV的AbsDiff方法来计算差异  

            difference = difference.Threshold(80, 255, ThresholdTypes.Binary);
            for (var i = 0; i < difference.Width; i++)
            {
                int mcount = 0;
                for (var j = 0; j < difference.Height; j++)
                {
                    var p = difference.Get<Vec3b>(j, i);
                    if (p.Item2 != 0)
                    {
                        mcount += 1;
                    }
                    if (mcount >= 5 && start_y == 0)
                    {
                        start_y = j - 5;
                    }
                }
                if (mcount > 5)
                {
                    start_x = i + 2;
                    break;
                }
            }
            Point point = new(start_x, start_y);
            return point;
        }

        /// <summary>
        /// 滑块位置识别算法2
        /// </summary>
        /// <param name="targetMat">滑块图片</param>
        /// <param name="backgroundMat">带缺口背景图片</param>
        /// <param name="simpleTarget"></param>
        /// <param name="flag">报错标识</param>
        /// <returns></returns>

        public static (int, Rect) SlideMatch(Mat targetMat, Mat backgroundMat, bool simpleTarget = false, bool flag = false)
        {
            Mat target;
            int target_y = 0;
            Point targetPoint = default;

            if (!simpleTarget)
            {
                try
                {
                    // Assuming GetTarget is a method that returns a Bitmap, targetX and targetY
                    (target, targetPoint) = GetTarget(targetMat);
                }
                catch (Exception)
                {
                    if (flag)
                    {
                        throw;
                    }
                    return SlideMatch(targetMat, backgroundMat, true, true);
                }
            }
            else
            {
                target = Cv2.ImDecode(targetMat.ImEncode(), ImreadModes.AnyColor);
            }

            Mat background = Cv2.ImDecode(backgroundMat.ImEncode(), ImreadModes.AnyColor);

            if (targetPoint != default)
            {
                target_y = targetPoint.Y;
                background = background.Clone(new Rect(0, target_y - 1, backgroundMat.Width, backgroundMat.Height - target_y + 1));
            }
            Mat cbackground = new();
            Mat ctarget = new();
            Cv2.Canny(background, cbackground, 100, 200);
            Cv2.Canny(target, ctarget, 100, 200);

            Cv2.CvtColor(cbackground, background, ColorConversionCodes.GRAY2BGR);
            Cv2.CvtColor(ctarget, target, ColorConversionCodes.GRAY2BGR);

            Mat res = new();
            Cv2.MatchTemplate(background, target, res, TemplateMatchModes.CCoeffNormed);
            Cv2.MinMaxLoc(res, out _, out _, out _, out Point maxLoc);
            var mRect = new Rect(maxLoc, target.Size());
            if (targetPoint != default)
            {
                mRect.Add(new Size(mRect.Width, target_y - 1));
            }
            return (targetPoint.Y, mRect);
        }


        /// <summary>
        /// 获取滑块精确方块内容
        /// </summary>
        /// <param name="image">滑块图片字节集</param>
        /// <returns></returns>
        public static (Mat, Point) GetTarget(Mat image)
        {

            //using (var ms = new MemoryStream(imgBytes))
            //{
            //    var image = Mat.FromStream(ms, ImreadModes.AnyColor);

            int w = image.Width, h = image.Height;
            int starttx = 0, startty = 0, endX = 0, endY = 0, endynext = 0, endxnext = 0;

            for (int x = 0; x < w; x++)
            {
                for (int y = 0; y < h; y++)
                {
                    var p = image.Get<Vec3b>(y, x);

                    if (p.Item2 == 0)
                    {
                        if (startty != 0 && endY < endynext)
                        {
                            endY = endynext;
                        }

                        if (starttx != 0 && endxnext > endX)
                        {
                            endX = endxnext;
                        }
                    }
                    else
                    {
                        if (startty == 0)
                        {
                            startty = y;

                            endY = 0;
                        }
                        else
                        {
                            if (y < startty)
                            {
                                startty = y;
                                endynext = 0;
                                endY = 0;
                            }
                            else
                            {

                                endynext = y;
                            }
                        }
                        if (starttx != 0 && endxnext < x)
                        {
                            endxnext = x;
                        }

                    }

                }
                if (starttx == 0 && startty != 0)
                {
                    starttx = x;
                }

                if (endY != 0 && endxnext > endX)
                {
                    endX = endxnext;
                }
            }

            var rect = new Rect(starttx, startty, endX - starttx, endY - startty);
            return (image.Clone(rect), new Point(starttx, startty));
            //}
        }
        #endregion

    }
}
