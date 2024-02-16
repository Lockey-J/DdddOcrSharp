using DdddOcrSharp;
using OpenCvSharp;
using System;
namespace DdddOcr.Demo
{
    internal class Program
    {
        static void Main(string[] args)
        {
            //Mat tg = new("E:\\slide.png", ImreadModes.AnyColor);
            //Mat bg = new("E:\\bg.png", ImreadModes.AnyColor);
            Mat image = new("det1.png", ImreadModes.AnyColor);
            Mat bg = new("det.png", ImreadModes.AnyColor);
            DDDDOCR ddddOcr = new(DdddOcrMode.Detect);

            var result = ddddOcr.Detect(bg.ToBytes());
          
           
            //Cv2.ImShow(new Random().Next(20).ToString(), target);
            foreach (var item in result)
            {
                //var mbg = bg.Clone(item);
                //Cv2.ImShow(new Random().Next(20).ToString(), mbg);               
                bg.Rectangle(item,new Scalar(0,0,0),1);
            }
            Cv2.ImShow("tg",bg);
            Cv2.WaitKey(0);
        }
    }
}
