import { useRef, useEffect, useState } from "react";
import { Camera, X, Image as ImageIcon, Scan } from "lucide-react";

type CameraCaptureProps = {
  onCapture: (imageData: string) => void;
  onLiveScan: () => void;
  onClose: () => void;   // ← App에서 받는 prop
};

export function CameraCapture({ onCapture, onLiveScan, onClose }: CameraCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(document.createElement("canvas"));
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isCameraReady, setIsCameraReady] = useState(false);

  useEffect(() => {
    startCamera();
    return () => stopCamera();
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsCameraReady(true);
      }
    } catch (err) {
      alert("카메라를 사용할 수 없습니다.");
      onClose();
    }
  };

  const stopCamera = () => {
    const stream = videoRef.current?.srcObject as MediaStream | null;
    stream?.getTracks().forEach((t) => t.stop());
  };

  const capturePhoto = () => {
    const v = videoRef.current;
    if (!v) return;

    const canvas = canvasRef.current;
    canvas.width = v.videoWidth;
    canvas.height = v.videoHeight;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(v, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/jpeg");

    stopCamera();
    onCapture(dataUrl);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onloadend = () => {
      stopCamera();
      onCapture(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  return (
    <div className="fixed inset-0 bg-black text-white flex flex-col">
      {/* 상단 닫기 버튼 */}
      <div className="absolute top-4 left-4 z-20">
        <button onClick={onClose} className="bg-black/60 p-2 rounded-full">
          <X className="w-6 h-6" />
        </button>
      </div>

      {/* 카메라 미리보기 */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="w-full h-full object-cover"
      />

      <canvas ref={canvasRef} className="hidden" />

      {/* 하단 촬영 UI */}
      {isCameraReady && (
        <div className="absolute bottom-0 left-0 right-0 pb-8 pt-4 bg-black/50 flex flex-col items-center space-y-4">

          <button
            onClick={capturePhoto}
            className="w-20 h-20 bg-white rounded-full border-4 border-gray-300 shadow-lg flex items-center justify-center"
          >
            <Camera className="w-10 h-10 text-black" />
          </button>

          <div className="flex gap-4">
            <button
              onClick={() => fileInputRef.current?.click()}
              className="bg-white/20 px-4 py-2 rounded-xl flex items-center gap-2"
            >
              <ImageIcon className="w-5 h-5" />
              갤러리
            </button>

            <button
              onClick={onLiveScan}
              className="bg-blue-600 px-4 py-2 rounded-xl flex items-center gap-2"
            >
              <Scan className="w-5 h-5" />
              실시간 스캔
            </button>
          </div>
        </div>
      )}

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handleFileSelect}
      />
    </div>
  );
}
