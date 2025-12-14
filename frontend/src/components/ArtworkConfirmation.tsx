import { useState, useEffect } from "react";
import { Check, X, Search, Loader2 } from "lucide-react";
import { Artwork } from "../App";

type Props = {
  capturedImage: string;
  onConfirm: (art: Artwork) => void;
  onRetake: () => void;
  onSearchByName: () => void;
};

const API_BASE = "https://localhost:8001";

// Base64 → Blob 변환
async function dataURLtoBlob(dataURL: string) {
  const res = await fetch(dataURL);
  return await res.blob();
}

export function ArtworkConfirmation({
  capturedImage,
  onConfirm,
  onRetake,
  onSearchByName,
}: Props) {
  const [loading, setLoading] = useState(true);
  const [artwork, setArtwork] = useState<Artwork | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    (async () => {
      setLoading(true);
      setError("");

      try {
        // 1) Base64 → FormData
        const blob = await dataURLtoBlob(capturedImage);
        const fd = new FormData();
        fd.append("file", blob, "capture.jpg");

        // 2) 이미지 매칭 요청
        const res = await fetch(`${API_BASE}/api/image/upload?topk=1`, {
          method: "POST",
          body: fd,
        });

        if (!res.ok) throw new Error("이미지 분석 실패");
        const data = await res.json();

        const top1 = data.results?.[0];
        if (!top1?.objectID) throw new Error("작품을 찾지 못했습니다.");

        // 3) 메타데이터 조회
        const metaRes = await fetch(`${API_BASE}/api/artwork/${top1.objectID}`);
        if (!metaRes.ok) throw new Error("작품 정보를 불러오지 못했습니다.");

        const meta = await metaRes.json();

        // 4) Artwork 타입으로 변환 (replace() 안전 처리)
        const converted: Artwork = {
          id: String(meta.id),
          title: meta.title ?? "제목 없음",
          artist: meta.artist ?? "작가 정보 없음",
          year: meta.year ?? "",
          description: meta.description ?? "",
          imageUrl: `${API_BASE}/images/${meta.id}.jpg`,   // replace 필요 없음
        };



        setArtwork(converted);
      } catch (e: any) {
        setError(e.message || "분석 중 오류가 발생했습니다.");
      } finally {
        setLoading(false);
      }
    })();
  }, [capturedImage]);

  // ---------------------------
  // 로딩 화면
  // ---------------------------
  if (loading) {
    return (
      <div className="flex flex-col h-screen items-center justify-center">
        <Loader2 className="w-14 h-14 text-blue-600 animate-spin mb-4" />
        <p className="text-gray-600">작품 분석 중입니다...</p>
      </div>
    );
  }

  // ---------------------------
  // 에러 화면
  // ---------------------------
  if (error) {
    return (
      <div className="flex flex-col h-screen p-6">
        <p className="text-red-600 font-semibold mb-2">오류 발생</p>
        <p className="text-gray-700 mb-6">{error}</p>

        <div className="flex gap-3">
          <button
            onClick={onRetake}
            className="flex-1 bg-gray-200 py-3 rounded-lg"
          >
            다시 촬영
          </button>

          <button
            onClick={onSearchByName}
            className="flex-1 bg-blue-600 text-white py-3 rounded-lg"
          >
            작품명으로 검색
          </button>
        </div>
      </div>
    );
  }

  // ---------------------------
  // 정상 화면
  // ---------------------------
  return (
    <div className="flex flex-col h-screen bg-white">
      {/* Header */}
      <div className="p-4 text-center shadow-sm bg-white">
        <h1 className="text-lg font-semibold">작품 확인</h1>
        <p className="text-gray-600 text-sm mt-1">이 작품이 맞나요?</p>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6 flex items-center justify-center">
        {artwork && (
          <div className="w-full flex flex-col items-center">

            {/* 🔥 원본 이미지만 유동적으로 확대/축소 */}
            <img
              src={artwork.imageUrl}
              alt={artwork.title}
              className="w-full max-w-4xl h-auto object-contain"
            />

            {/* 작품 정보 */}
            <div className="w-full max-w-4xl mt-6 px-4">
              <h2 className="text-2xl font-bold">{artwork.title}</h2>
              <p className="text-gray-700 mt-1 text-lg">{artwork.artist}</p>
              <p className="text-gray-500">{artwork.year}</p>
            </div>

          </div>
        )}
      </div>



      {/* Bottom Buttons */}
      <div className="p-6 border-t bg-white space-y-3">
        <div className="flex gap-3">
          <button
            onClick={onRetake}
            className="flex-1 flex items-center justify-center bg-gray-200 py-3 rounded-xl text-gray-800"
          >
            <X className="w-5 h-5 mr-1" /> 다시 촬영
          </button>

          <button
            onClick={() => artwork && onConfirm(artwork)}
            className="flex-1 flex items-center justify-center bg-blue-600 py-3 rounded-xl text-white"
          >
            <Check className="w-5 h-5 mr-1" /> 확인
          </button>
        </div>

        <button
          onClick={onSearchByName}
          className="flex items-center justify-center w-full border border-blue-600 text-blue-600 py-3 rounded-xl"
        >
          <Search className="w-5 h-5 mr-2" />
          작품이 아닌가요? 작품명으로 검색하기
        </button>
      </div>
    </div>
  );
}
