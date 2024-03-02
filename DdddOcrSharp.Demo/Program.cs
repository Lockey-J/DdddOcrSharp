using DdddOcrSharp;
using OpenCvSharp;
using System;
using System.Text.Json;

namespace DdddOcr.Demo
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Mat tg = new("tg3.png", ImreadModes.AnyColor);
            Mat bg = new("bg3.png", ImreadModes.AnyColor);
            Mat ocr = new("ocr.jpg", ImreadModes.AnyColor);
            Mat det = new("det.png", ImreadModes.AnyColor);

            DDDDOCR ddddOcrDet = new(DdddOcrMode.Detect);
            DDDDOCR ddddOcrOcrOld = new(DdddOcrMode.ClassifyOld);
            DDDDOCR ddddOcrOcrNew = new(DdddOcrMode.ClassifyBeta);



            var OcrOldResult = ddddOcrOcrOld.Classify(ocr.ToBytes());
            Console.WriteLine("旧版本文本识别结果：" + OcrOldResult);
            Console.WriteLine("\r\n");

            var OcrNewResult = ddddOcrOcrNew.Classify(ocr.ToBytes());
            Console.WriteLine("新版本文本识别结果：" + OcrNewResult);
            Console.WriteLine("\r\n");

            var Detresult = ddddOcrDet.Detect(det.ToBytes());
            foreach (var item in Detresult)
            {
                det.Rectangle(item, new Scalar(0, 0, 255), 2);
            }

            Cv2.ImShow("det", det);
            Console.WriteLine("目标识别到的坐标为：" + JsonSerializer.Serialize(Detresult));
            Console.WriteLine("\r\n");

            var (target_y, rect) = DDDDOCR.SlideMatch(tg, bg);
            Console.WriteLine("SlideMatch滑块的Y坐标为：" + target_y + "\r\nSlideMatch滑块缺口方框为:" + JsonSerializer.Serialize<Rect>(rect));
            bg.Rectangle(rect, new Scalar(0, 0, 255), 2);
            Cv2.ImShow("SlideMatch", bg);
            Console.WriteLine("\r\n");
            Cv2.WaitKey(0);
        }
    }
}
