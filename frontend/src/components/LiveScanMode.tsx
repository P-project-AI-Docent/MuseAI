import { useState, useRef, useEffect } from 'react';
import { BookOpen, MessageCircle } from 'lucide-react';
import { Artwork } from '../App';

type LiveScanModeProps = {
  onMatch: (artwork: Artwork, frameData: string, mode: 'guide' | 'chat') => void;
  onBack: () => void;
};

const API_BASE = "https://localhost:8001";

export function LiveScanMode({ onMatch, onBack }: LiveScanModeProps) {
  const [detectedArtwork, setDetectedArtwork] = useState<Artwork | null>(null);
  const [detectedFrame, setDetectedFrame] = useState<string | null>(null);
  const [isDetecting, setIsDetecting] = useState(true);

  const [isTTSLoading, setIsTTSLoading] = useState(false);
  const [isTTSPlaying, setIsTTSPlaying] = useState(false);
  const [isAudioReady, setIsAudioReady] = useState(false);

  const [toastMsg, setToastMsg] = useState("");

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(document.createElement("canvas"));

  const scanRef = useRef<number | null>(null);
  const rafRef = useRef<number | null>(null);

  /* -----------------------------------------
     토스트
  ----------------------------------------- */
  const showToast = (msg: string) => {
    setToastMsg(msg);
    setTimeout(() => setToastMsg(""), 2000);
  };

  /* -----------------------------------------
     카메라 시작/정지
  ----------------------------------------- */
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
      }
    } catch {
      alert("카메라 권한이 필요합니다.");
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
  };

  /* -----------------------------------------
     프레임 캡처
  ----------------------------------------- */
  const captureFrameDataURL = () => {
    const v = videoRef.current;
    const c = canvasRef.current;
    if (!v || !v.videoWidth || !v.videoHeight) return null;

    c.width = v.videoWidth;
    c.height = v.videoHeight;

    const ctx = c.getContext("2d");
    if (!ctx) return null;

    ctx.drawImage(v, 0, 0);
    return c.toDataURL("image/jpeg");
  };

  const dataURLtoBlob = async (dataURL: string) =>
    (await fetch(dataURL)).blob();

  /* -----------------------------------------
     작품 인식 (🔥 function 선언으로 변경!)
  ----------------------------------------- */
  async function tryDetect() {
    if (!isDetecting) return;

    const frameData = captureFrameDataURL();
    if (!frameData) return;

    try {
      const blob = await dataURLtoBlob(frameData);
      const fd = new FormData();
      fd.append("file", blob, "frame.jpg");
      fd.append("topk", "1");

      const res1 = await fetch(`${API_BASE}/api/image/upload`, {
        method: "POST",
        body: fd,
      });

      if (!res1.ok) return;

      const j1 = await res1.json();
      const hit = j1.results?.[0];
      if (!hit?.objectID) return;

      const res2 = await fetch(`${API_BASE}/api/artwork/${hit.objectID}`);
      if (!res2.ok) return;
      const meta = await res2.json();

      setDetectedArtwork({
        id: meta.id,
        title: meta.title,
        artist: meta.artist,
        year: meta.year,
        description: meta.description,
        imageUrl: `${API_BASE}/images/${meta.id}.jpg`,
      });

      setDetectedFrame(frameData);
      setIsDetecting(false);

    } catch (err) {
      console.error(err);
    }
  }

  /* -----------------------------------------
     초기 스캔 시작
  ----------------------------------------- */
  useEffect(() => {
    startCamera();
    scanRef.current = window.setInterval(() => tryDetect(), 700);

    return () => {
      if (scanRef.current) clearInterval(scanRef.current);
      stopCamera();
    };
  }, []);

  /* -----------------------------------------
     stopScanning 이벤트 → 즉시 종료
  ----------------------------------------- */
  useEffect(() => {
    const stopHandler = () => {
      if (scanRef.current) {
        clearInterval(scanRef.current);
        scanRef.current = null;
      }
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      stopCamera();
      setIsDetecting(false);
    };

    window.addEventListener("stopScanning", stopHandler);
    return () => window.removeEventListener("stopScanning", stopHandler);
  }, []);

  /* -----------------------------------------
     전체 설명 준비
  ----------------------------------------- */
  const prepareFullDescription = async () => {
    window.dispatchEvent(new Event("stopScanning"));
    if (!detectedArtwork) return;

    setIsTTSLoading(true);
    setIsAudioReady(false);

    try {
      const res = await fetch(`${API_BASE}/api/artwork/${detectedArtwork.id}/full-description`);
      const data = await res.json();

      const blobRes = await fetch(`${API_BASE}${data.audioUrl}`);
      const audioBlob = await blobRes.blob();
      const audioUrl = URL.createObjectURL(audioBlob);

      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.src = "";
      }

      const audio = new Audio();
      audio.src = audioUrl;
      audioRef.current = audio;

      audio.onplay = () => setIsTTSPlaying(true);
      audio.onended = () => {
        setIsTTSPlaying(false);
        showToast("전체 설명이 끝났어요!");
      };

      audio.load();
      setIsAudioReady(true);

    } catch {
      alert("전체 설명을 준비하지 못했습니다.");
    } finally {
      setIsTTSLoading(false);
    }
  };

  const playAudio = () => audioRef.current?.play();

  const stopTTS = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    setIsTTSPlaying(false);
  };

  /* -----------------------------------------
     guide/chat 전환
  ----------------------------------------- */
  const handleConfirm = (mode: 'guide' | 'chat') => {
    window.dispatchEvent(new Event("stopScanning"));

    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    setIsAudioReady(false);
    setIsTTSPlaying(false);

    if (!detectedArtwork || !detectedFrame) return;
    onMatch(detectedArtwork, detectedFrame, mode);
  };

  /* -----------------------------------------
     UI
  ----------------------------------------- */
  return (
    <div className="flex flex-col h-screen bg-black">

      {toastMsg && (
        <div className="absolute top-5 left-1/2 -translate-x-1/2 bg-white px-4 py-2 rounded-xl shadow">
          {toastMsg}
        </div>
      )}

      <div className="bg-gradient-to-r from-blue-600 to-cyan-600 text-white p-4 flex items-center gap-3">
        <button onClick={onBack} className="p-2 hover:bg-white/20 rounded-lg">←</button>
        <div>
          <h2>실시간 스캔 모드</h2>
          <p className="text-cyan-100 text-sm">작품에 카메라를 가까이 대주세요</p>
        </div>
      </div>

      <div className="flex-1 relative bg-black">
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className="w-full h-full object-cover"
        />

        {isDetecting && (
          <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-black/70 text-white px-4 py-2 rounded-full">
            작품 인식 중...
          </div>
        )}

        {!isDetecting && detectedArtwork && (
          <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black via-black/90 to-transparent">
            <div className="bg-white rounded-2xl shadow-xl max-w-md mx-auto overflow-hidden">

              <img src={detectedArtwork.imageUrl} className="w-full h-48 object-cover" />

              <div className="p-4">
                <h3 className="text-blue-900">{detectedArtwork.title}</h3>
                <p className="text-gray-600">
                  {detectedArtwork.artist}
                  {detectedArtwork.year ? ` · ${detectedArtwork.year}` : ""}
                </p>

                <p className="text-gray-700 mt-2 line-clamp-3">
                  {detectedArtwork.description}
                </p>

                {/* 🔵 준비중 */}
                {isTTSLoading && (
                  <div className="mt-3 text-blue-600 text-center">
                    전체 설명 준비 중...
                  </div>
                )}

                {/* 🔵 재생 중 */}
                {isTTSPlaying && (
                  <button
                    onClick={stopTTS}
                    className="mt-3 w-full py-3 bg-red-100 text-red-600 rounded-xl"
                  >
                    🔊 재생 중지
                  </button>
                )}

                {/* 🔵 준비 버튼 */}
                <button
                  onClick={prepareFullDescription}
                  className="w-full mt-4 py-3 bg-gradient-to-r from-blue-600 to-cyan-600 text-white rounded-xl flex items-center justify-center gap-2"
                >
                  <BookOpen className="w-5 h-5" />
                  전체 설명
                </button>

                {/* 🔵 재생 버튼 */}
                {isAudioReady && !isTTSPlaying && (
                  <button
                    onClick={playAudio}
                    className="w-full mt-3 py-3 bg-blue-600 text-white rounded-xl"
                  >
                    ▶ 음성 재생
                  </button>
                )}

                <div className="flex gap-3 mt-4">
                  <button
                    onClick={() => {
                      setIsDetecting(true);
                      setDetectedArtwork(null);
                      setDetectedFrame(null);

                      startCamera();
                      if (scanRef.current) clearInterval(scanRef.current);
                      scanRef.current = window.setInterval(() => tryDetect(), 700);
                    }}
                    className="flex-1 bg-gray-100 py-3 rounded-xl"
                  >
                    다시 스캔
                  </button>

                  <button
                    onClick={() => handleConfirm("chat")}
                    className="flex-1 border-2 border-blue-600 text-blue-600 py-3 rounded-xl flex items-center justify-center gap-2"
                  >
                    <MessageCircle className="w-5 h-5" />
                    질의응답
                  </button>
                </div>

              </div>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
