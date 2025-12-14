// src/api/index.ts
const API_BASE = (import.meta.env.VITE_API_URL ?? "https://localhost:8001").replace(/\/$/, "");
export const API = `${API_BASE}/api`;

// 변환 함수
export function convertToArtwork(r: any) {
  return {
    id: String(r.objectID),
    title: r.title || "제목 없음",
    artist: r.artistDisplayName || r.artist || "작가 정보 없음",
    year: r.objectDate || "",
    description: r.met_description || "",
    imageUrl: r.imageUrl
      ? r.imageUrl
      : `${API_BASE}/images/${(r.image_path || "").split("/").pop()}`,
  };
}

export async function uploadArtworkImage(file: File) {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${API}/image/upload?topk=1`, {
    method: "POST",
    body: form,
  });

  const data = await res.json();
  return convertToArtwork(data.results[0]);
}

export async function uploadArtworkBase64(base64: string) {
  const blob = await fetch(base64).then((r) => r.blob());
  const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
  return uploadArtworkImage(file);
}

export async function searchByText(query: string) {
  const res = await fetch(`${API}/text/search?q=${encodeURIComponent(query)}`);
  const data = await res.json();
  return (data.results || []).map(convertToArtwork);
}

export async function sendChat(objectID: string, question: string, style: string) {
  const res = await fetch(`${API}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      objectID: Number(objectID),
      question,
      style,
      sessionId: "mobile-user",
    }),
  });

  return res.json();
}
