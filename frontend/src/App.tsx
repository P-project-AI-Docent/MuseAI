import { useState, useRef, useEffect } from "react";
import { CameraCapture } from "./components/CameraCapture";
import { ArtworkConfirmation } from "./components/ArtworkConfirmation";
import { ArtworkChat } from "./components/ArtworkChat";
import { AuthScreen } from "./components/AuthScreen";
import { ProfileScreen } from "./components/ProfileScreen";
import { BottomNav } from "./components/BottomNav";
import { ArtworkSearch } from "./components/ArtworkSearch";
import { WelcomeScreen } from "./components/WelcomeScreen";
import { LiveScanMode } from "./components/LiveScanMode";
import { DocentVersionSelector } from "./components/DocentVersionSelector";

// =============================
// Artwork 타입
// =============================
export type Artwork = {
  id: string;
  title: string;
  artist: string;
  year: string;
  description: string;
  imageUrl: string;
  mode?: "camera" | "qna"; 
};

export type DocentVersion = "general" | "child" | "expert";

type AppState =
  | "auth"
  | "versionselect"
  | "welcome"
  | "camera"
  | "cameraOn"
  | "confirmation"
  | "chat"
  | "profile"
  | "search"
  | "livescan";

type User = {
  email: string;
  name: string;
  docentVersion?: DocentVersion;
} | null;

export default function App() {
  const [user, setUser] = useState<User>(() => {
    const saved = localStorage.getItem("currentUser");
    return saved ? JSON.parse(saved) : null;
  });

  const [hasSeenWelcome, setHasSeenWelcome] = useState(() => {
    return localStorage.getItem("hasSeenWelcome") === "true";
  });

  const [state, setState] = useState<AppState>(() => {
    if (!user) return "auth";
    if (!user.docentVersion) return "versionselect";
    if (!hasSeenWelcome) return "welcome";
    return "camera";
  });

  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [selectedArtwork, setSelectedArtwork] = useState<Artwork | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  

  // =======================================================
  // 유사 작품 → 새로운 QnA
  // =======================================================
  useEffect(() => {
    const handler = (e: any) => {
      const item = e.detail;

      setSelectedArtwork({
        id: String(item.objectID),
        title: item.title,
        artist: item.artist,
        year: "",
        description: "",
        imageUrl: item.imageUrl,
        mode: "qna"    // 🔥 유사 작품에서 넘어온 경우
      });

      setState("chat");
    };

    window.addEventListener("openArtworkFromChat", handler);
    return () => window.removeEventListener("openArtworkFromChat", handler);
  }, []);

  // =======================================================
  // 🔥 전체 설명 / 또는 채팅 들어가면 스캔 완전 정지
  // =======================================================
  useEffect(() => {
    const stopHandler = () => {
      console.log("📌 stopScanning 이벤트 수신 → 스캔 중지");

      // LiveScanMode에서 쓰고 있는 scanning state 끔
      setState((prev) => {
        // 만약 livescan에 있었으면 camera로 돌려보내도 됨
        // 단, 현재 흐름에서는 단순히 state 강제 이동 없이 scanning만 멈추면 됨
        return prev;
      });

      // 카메라 스트림 정지
      const video = document.querySelector("video");
      if (video?.srcObject) {
        const tracks = (video.srcObject as MediaStream).getTracks();
        tracks.forEach((t) => t.stop());
      }
    };

    window.addEventListener("stopScanning", stopHandler);
    return () => window.removeEventListener("stopScanning", stopHandler);
  }, []);


  // =======================================================
  // 갤러리 업로드
  // =======================================================
  const handleGalleryUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onloadend = () => {
      setCapturedImage(reader.result as string);
      setState("confirmation");
    };
    reader.readAsDataURL(file);
  };

  // =======================================================
  // 로그인 / 버전 선택 / 웰컴
  // =======================================================
  const handleLogin = (info: { email: string; name: string }) => {
    const savedVersion = localStorage.getItem(
      `docentVersion_${info.email}`
    ) as DocentVersion | null;

    const newUser = {
      ...info,
      docentVersion: savedVersion ?? undefined,
    };

    setUser(newUser);
    localStorage.setItem("currentUser", JSON.stringify(newUser));

    if (!savedVersion) setState("versionselect");
    else setState("welcome");
  };

  const handleVersionSelect = (version: DocentVersion) => {
    if (!user) return;

    const updated = { ...user, docentVersion: version };
    setUser(updated);

    localStorage.setItem("currentUser", JSON.stringify(updated));
    localStorage.setItem(`docentVersion_${user.email}`, version);
  };

  const handleWelcomeComplete = () => {
    setHasSeenWelcome(true);
    localStorage.setItem("hasSeenWelcome", "true");
    setState("camera");
  };

  // =======================================================
  // 로그아웃
  // =======================================================
  const handleLogout = () => {
    if (!user) return;

    localStorage.removeItem(`history_${user.email}`);
    localStorage.removeItem(`docentVersion_${user.email}`);
    localStorage.removeItem("currentUser");
    localStorage.removeItem("hasSeenWelcome");

    setUser(null);
    setState("auth");
  };

  // =======================================================
  // 촬영 이미지
  // =======================================================
  const handleImageCapture = (img: string) => {
    setCapturedImage(img);
    setState("confirmation");
  };

  const handleConfirm = (artwork: Artwork) => {
    setSelectedArtwork({
      ...artwork,
      mode: "camera"   // 🔥 촬영해서 확인한 작품임을 기록
    });
    setState("chat");
  };


  const handleRetake = () => {
    setCapturedImage(null);
    setSelectedArtwork(null);
    setState("camera");
  };

  const handleBackToCamera = () => {
    setCapturedImage(null);
    setSelectedArtwork(null);
    setState("camera");
  };

  // =======================================================
  // 히스토리 저장
  // =======================================================
  useEffect(() => {
    if (state !== "chat") return;
    if (!user || !selectedArtwork) return;

    const key = `history_${user.email}`;
    const old = JSON.parse(localStorage.getItem(key) || "[]");

    const newEntry = {
      artworkId: selectedArtwork.id,
      title: selectedArtwork.title,
      artist: selectedArtwork.artist,
      imageUrl: selectedArtwork.imageUrl,
      timestamp: Date.now(),
      mode: selectedArtwork.mode || "camera"   // 🔥 mode 반영
    };

    const updated = [...old, newEntry];

    localStorage.setItem(key, JSON.stringify(updated));
  }, [state, selectedArtwork, user]);


  // =======================================================
  // 렌더링
  // =======================================================
  return (
    <div className="min-h-screen w-full bg-gray-50 overflow-hidden">
      {state === "auth" && <AuthScreen onLogin={handleLogin} />}

      {state === "versionselect" && (
        <DocentVersionSelector userName={user!.name} onSelect={handleVersionSelect} />
      )}

      {state === "welcome" && (
        <WelcomeScreen userName={user!.name} onStart={handleWelcomeComplete} />
      )}

      {/*   초기 카메라 시작 화면   */}
      {state === "camera" && (
        <div className="flex flex-col h-screen w-full bg-gray-50">
          <div className="text-center pt-10 pb-6">
            <h1 className="text-lg font-semibold">AI 도슨트</h1>
            <p className="text-gray-600">작품을 촬영하여 정보를 확인하세요</p>
            <p className="font-bold text-blue-600 mt-1">{user!.name}님 환영합니다</p>
          </div>

          <div className="flex-1 flex items-center justify-center w-full">
            <div className="w-full h-[58vh] bg-black flex flex-col items-center justify-center">
              <div className="text-6xl text-gray-300 mb-4">📷</div>
              <p className="text-gray-200 mb-4">카메라를 시작하세요</p>

              <button
                onClick={() => setState("cameraOn")}
                className="px-6 py-3 bg-blue-600 text-white rounded-xl shadow"
              >
                카메라 시작
              </button>
            </div>
          </div>

          <div className="px-6 pb-24 space-y-3 w-full">
            <button
              onClick={() => fileInputRef.current?.click()}
              className="w-full p-3 bg-gray-200 rounded-xl"
            >
              갤러리
            </button>

            <input
              type="file"
              accept="image/*"
              ref={fileInputRef}
              className="hidden"
              onChange={handleGalleryUpload}
            />

            <button
              onClick={() => setState("livescan")}
              className="w-full p-3 rounded-xl bg-gradient-to-r from-blue-600 to-cyan-600 text-white"
            >
              실시간 스캔 모드
            </button>
          </div>

          <BottomNav currentPage="camera" onNavigate={(p) => setState(p)} />
        </div>
      )}

      {/* 카메라 화면 */}
      {state === "cameraOn" && (
        <CameraCapture
          onCapture={handleImageCapture}
          onLiveScan={() => setState("livescan")}
          onClose={() => setState("camera")}
        />
      )}

      {/* 프로필 */}
      {state === "profile" && (
        <>
          <ProfileScreen
            user={user!}
            onLogout={handleLogout}
            onVersionChange={handleVersionSelect}
          />
          <BottomNav currentPage="profile" onNavigate={(p) => setState(p)} />
        </>
      )}

      {/* 촬영 후 작품 확인 */}
      {state === "confirmation" && capturedImage && (
        <ArtworkConfirmation
          capturedImage={capturedImage}
          onConfirm={handleConfirm}
          onRetake={handleRetake}
          onSearchByName={() => setState("search")}
        />
      )}

      {/* 검색 */}
      {state === "search" && (
        <ArtworkSearch
          onSelect={(art) => {
            setSelectedArtwork(art);
            setState("chat");
          }}
          onBackToCamera={handleBackToCamera}
        />
      )}

      {/* 실시간 스캔 모드 */}
      {state === "livescan" && (
        <LiveScanMode
          onBack={handleBackToCamera}
          onMatch={(artwork, frameData, mode) => {

            // frameData 절대 사용 금지
            setSelectedArtwork({
              ...artwork,
              imageUrl: artwork.imageUrl,
            });

            // 🔥 guide 모드 → 화면 유지(이동 금지)
            if (mode === "guide") return;

            // 🔥 chat 모드 → 채팅 화면 이동
            if (mode === "chat") {
              setState("chat");
            }
          }}
        />
      )}

      {/* 채팅 */}
      {state === "chat" && selectedArtwork && (
        <ArtworkChat
          key={selectedArtwork.id}
          artwork={selectedArtwork}
          onBackToCamera={handleBackToCamera}
          docentVersion={user!.docentVersion!}
        />
      )}
    </div>
  );
}
