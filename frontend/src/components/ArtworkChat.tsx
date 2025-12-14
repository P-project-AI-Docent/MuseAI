import { useState, useRef, useEffect } from "react";
import {
  Send,
  ArrowLeft,
  Volume2,
  VolumeX,
  BookOpen,
  Mic,
  MicOff,
  Loader2,
} from "lucide-react";

import { Artwork, DocentVersion } from "../App";

type Message = {
  id: string;
  role: "user" | "assistant" | "similar" | "system";
  content?: string;
  timestamp: Date;
  similarResults?: SimilarArtwork[];
};

type SimilarArtwork = {
  objectID: number;
  title: string;
  artist: string;
  imageUrl: string;
};

type Props = {
  artwork: Artwork;
  onBackToCamera: () => void;
  docentVersion: DocentVersion;
};

const API_BASE = "https://localhost:8001";

function mapStyle(v: DocentVersion): "docent" | "kids" | "expert" {
  if (v === "child") return "kids";
  if (v === "expert") return "expert";
  return "docent";
}

// iOS STT 타입 처리
type WebSpeechRecognition = typeof window extends any
  ? (Window & typeof globalThis) & {
      webkitSpeechRecognition?: any;
      SpeechRecognition?: any;
    }
  : never;

export function ArtworkChat({ artwork, onBackToCamera, docentVersion }: Props) {
  const welcomeMessage =
    docentVersion === "child"
      ? `안녕! 나는 AI 도슨트야! "${artwork.title}"에 대해 궁금한 게 있으면 뭐든지 말해줘!`
      : docentVersion === "expert"
      ? `"${artwork.title}" 작품에 대한 전문 분석을 제공합니다. 질문이 있으신가요?`
      : `안녕하세요! "${artwork.title}"에 대해 궁금한 점을 편하게 물어보세요.`;

  // 채팅 메시지
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "m1",
      role: "assistant",
      content: welcomeMessage,
      timestamp: new Date(),
    },
  ]);

  const [inputValue, setInputValue] = useState("");
  const [isComposing, setIsComposing] = useState(false);

  const [isTyping, setIsTyping] = useState(false);

  const [autoSpeak, setAutoSpeak] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);

  const [askingCriterion, setAskingCriterion] = useState(false);

  const [similarResults, setSimilarResults] = useState<SimilarArtwork[]>([]);

  // 👇 새로 추가
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [isAudioReady, setIsAudioReady] = useState(false);
  const [isTTSLoading, setIsTTSLoading] = useState(false);

  const [isAudioPlaying, setIsAudioPlaying] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // STT 관련
  const [isListening, setIsListening] = useState(false);
  const recognitionRef = useRef<any | null>(null);

  const [toastMsg, setToastMsg] = useState("");
  const toastTimerRef = useRef<number | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const sessionIdRef = useRef<string>(
    `s_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
  );

  const showToast = (msg: string, ms = 1500) => {
    setToastMsg(msg);
    if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
    toastTimerRef.current = window.setTimeout(() => setToastMsg(""), ms);
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const speakText = (txt: string) => {
    const u = new SpeechSynthesisUtterance(txt);
    u.lang = "ko-KR";
    u.rate = 1.25;
    u.onstart = () => setIsSpeaking(true);
    u.onend = () => setIsSpeaking(false);
    window.speechSynthesis.speak(u);
  };

  const stopSpeaking = () => {
    window.speechSynthesis.cancel();
    setIsSpeaking(false);
  };

  const fetchMetasForResults = async (arr: { objectID: number }[]) => {
    const out: SimilarArtwork[] = [];
    for (const r of arr) {
      try {
        const res = await fetch(`${API_BASE}/api/artwork/${r.objectID}`);
        if (res.ok) {
          const meta = await res.json();
          out.push({
            objectID: Number(meta.id),
            title: meta.title || "",
            artist: meta.artist || "",
            imageUrl: `${API_BASE}/images/${meta.id}.jpg`,
          });
          continue;
        }
      } catch {}
      out.push({
        objectID: r.objectID,
        title: "",
        artist: "",
        imageUrl: `${API_BASE}/images/${r.objectID}.jpg`,
      });
    }
    return out;
  };

  // ===========================================================
  //  🔥 전체 설명 호출 (여기서는 오디오 객체 생성 금지)
  // ===========================================================
  const playFullDescription = async () => {
    try {
      window.dispatchEvent(new CustomEvent("stopScanning"));
      setIsTTSLoading(true);
      setIsAudioReady(false);
      setAudioBlob(null);

      const loaderId = `sys_${Date.now()}`;
      setMessages((prev) => [
        ...prev,
        {
          id: loaderId,
          role: "system",
          content: "전체 설명을 준비하고 있어요...",
          timestamp: new Date(),
        },
      ]);

      const res = await fetch(
        `${API_BASE}/api/artwork/${artwork.id}/full-description`
      );
      const data = await res.json();

      // 메시지 갱신
      setMessages((prev) =>
        prev
          .filter((m) => m.id !== loaderId)
          .concat({
            id: `full_${Date.now()}`,
            role: "assistant",
            content: data.text,
            timestamp: new Date(),
          })
      );

      // 🔥 오디오 blob 다운로드만 함 (여기서는 Audio() 만들면 iOS 차단!)
      const audioRes = await fetch(API_BASE + data.audioUrl);
      const blob = await audioRes.blob();
      setAudioBlob(blob);

      setIsAudioReady(true); // 재생 버튼 활성화

    } catch (e) {
      showToast("오디오를 불러오지 못했습니다.");
    } finally {
      setIsTTSLoading(false);
    }
  };

  // ===========================================================
  //  🔥 iOS 허용 방식: 재생 버튼 클릭 시 Audio() 생성
  // ===========================================================
  const handleAudioPlay = () => {
    if (!audioBlob) return;

    try {
      const url = URL.createObjectURL(audioBlob);

      // iOS 허용: 사용자 이벤트 안에서 Audio 객체 생성
      const audio = new Audio(url);
      audioRef.current = audio;

      audio.onplay = () => setIsAudioPlaying(true);
      audio.onended = () => setIsAudioPlaying(false);
      audio.onerror = () => showToast("오디오 재생 오류");

      audio.load();
      audio.play();
    } catch (e) {
      showToast("재생할 수 없습니다");
    }
  };

  const stopAudio = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    setIsAudioPlaying(false);
  };

  // ===========================================================
  //  STT
  // ===========================================================
  const initRecognition = () => {
    const w = window as unknown as WebSpeechRecognition;
    const SR = w.SpeechRecognition || w.webkitSpeechRecognition;
    if (!SR) {
      showToast("음성 인식을 사용할 수 없습니다.");
      return null;
    }
    const rec = new SR();
    rec.lang = "ko-KR";
    rec.continuous = true;
    rec.interimResults = true;
    return rec;
  };

  const startListening = () => {
    if (isListening) return;
    const rec = initRecognition();
    if (!rec) return;

    recognitionRef.current = rec;

    let finalTxt = "";

    rec.onstart = () => {
      showToast("듣는 중...");
      setIsListening(true);
    };

    rec.onresult = (ev: any) => {
      let interim = "";
      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        const r = ev.results[i];
        const t = r[0].transcript;
        if (r.isFinal) finalTxt += t;
        else interim += t;
      }
      setInputValue((finalTxt + " " + interim).trim());
    };

    rec.onerror = () => showToast("음성 인식 오류");
    rec.onend = () => {
      setIsListening(false);
      setTimeout(() => setInputValue(""), 100);
    };

    rec.start();
  };

  const stopListening = () => {
    if (!isListening) return;
    recognitionRef.current?.stop();
    recognitionRef.current = null;
  };

  const toggleMic = () => {
    if (isListening) stopListening();
    else startListening();
  };

  // ===========================================================
  // 메시지 전송
  // ===========================================================
  const sendQuestion = async (q: string) => {
    setIsTyping(true);
    setAskingCriterion(false);

    try {
      const payload = {
        question: q,
        objectID: Number(artwork.id),
        style: mapStyle(docentVersion),
        sessionId: sessionIdRef.current,
      };

      const res = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();

      const answer = data.answer ?? "(응답 없음)";
      setMessages((prev) => [
        ...prev,
        {
          id: `ai_${Date.now()}`,
          role: "assistant",
          content: answer,
          timestamp: new Date(),
        },
      ]);

      if (autoSpeak) speakText(answer);

      if (
        answer.includes("기준") ||
        answer.includes("유사한 작품") ||
        answer.includes("시각적") ||
        answer.includes("내용")
      ) {
        setAskingCriterion(true);
      }

      if (data.results && data.results.length > 0) {
        const metas = await fetchMetasForResults(
          data.results.map((x: any) => ({ objectID: Number(x.objectID) }))
        );
        setMessages((prev) => [
          ...prev,
          {
            id: `sim_${Date.now()}`,
            role: "similar",
            timestamp: new Date(),
            similarResults: metas,
          },
        ]);
      }
    } catch {
      showToast("오류가 발생했습니다.");
    } finally {
      setIsTyping(false);
    }
  };

  const handleSendMessage = (customText?: string) => {
    const raw = customText ?? inputValue;
    const q = raw.trim();
    if (!q) return;

    setInputValue("");
    requestAnimationFrame(() => setInputValue(""));

    setMessages((prev) => [
      ...prev,
      {
        id: `u_${Date.now()}`,
        role: "user",
        content: q,
        timestamp: new Date(),
      },
    ]);

    sendQuestion(q);
  };

  const openArtwork = (item: SimilarArtwork) => {
    window.dispatchEvent(
      new CustomEvent("openArtworkFromChat", { detail: item })
    );
  };

  // ===========================================================
  // UI 렌더링
  // ===========================================================
  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-blue-50 via-cyan-50 to-teal-50">
      {/* HEADER */}
      <div className="bg-gradient-to-r from-blue-600 to-cyan-600 text-white shadow-lg pb-5">
        <div className="p-4 flex items-center gap-3">
          <button onClick={onBackToCamera} className="p-2 hover:bg-white/20 rounded-lg">
            <ArrowLeft className="w-5 h-5" />
          </button>

          <div className="flex-1">
            <h2 className="font-semibold text-lg line-clamp-1">{artwork.title}</h2>
            <p className="text-blue-100 text-sm">{artwork.artist}</p>
          </div>

          <button
            onClick={() => setAutoSpeak(!autoSpeak)}
            className="p-2 hover:bg-white/20 rounded-lg"
          >
            {autoSpeak ? <Volume2 /> : <VolumeX />}
          </button>
        </div>

        <div className="px-4">
          <img
            src={artwork.imageUrl}
            className="w-full h-40 object-cover rounded-lg border border-white/30 shadow"
          />

          {/* 전체 설명 */}
          <button
            onClick={playFullDescription}
            disabled={isTTSLoading}
            className="mt-3 w-full py-3 bg-white/20 text-white rounded-lg flex items-center justify-center gap-2"
          >
            {isTTSLoading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                전체 설명 생성 중...
              </>
            ) : (
              <>
                <BookOpen className="w-5 h-5" />
                전체 도슨트 설명 듣기
              </>
            )}
          </button>

          {/* 🔊 재생 버튼 (iOS 친화적) */}
          {isAudioReady && (
            <button
              onClick={handleAudioPlay}
              className="mt-3 w-full py-3 bg-blue-600 text-white rounded-lg"
            >
              ▶ 음성 재생
            </button>
          )}

          {isAudioPlaying && (
            <div className="mt-3 w-full py-2 bg-white/10 text-white rounded-lg flex justify-between px-4">
              <span>음성 재생 중...</span>
              <button onClick={stopAudio} className="underline">
                중지
              </button>
            </div>
          )}
        </div>
      </div>

      {/* BODY */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((m) => {
          // 유사작품 카드
          if (m.role === "similar" && m.similarResults) {
            return (
              <div key={m.id} className="bg-white rounded-xl border p-3 space-y-5 shadow">
                <p className="text-base font-semibold">유사한 작품</p>
                {m.similarResults.map((it) => (
                  <button
                    key={it.objectID}
                    onClick={() => openArtwork(it)}
                    className="w-full text-left rounded-xl overflow-hidden bg-gray-50 border"
                  >
                    <img src={it.imageUrl} className="w-full max-h-[320px] object-contain" />
                    <div className="p-3">
                      <p className="text-lg font-semibold">{it.title || "제목 없음"}</p>
                      <p className="text-sm text-gray-600">{it.artist}</p>
                      <p className="text-blue-600 text-sm mt-2">QnA로 보기 →</p>
                    </div>
                  </button>
                ))}
              </div>
            );
          }

          // 시스템 메시지
          if (m.role === "system") {
            return (
              <div key={m.id} className="flex items-center gap-2 text-sm text-blue-700">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>{m.content}</span>
              </div>
            );
          }

          // 일반 메시지
          return (
            <div
              key={m.id}
              className={`flex gap-3 ${m.role === "user" ? "justify-end" : "justify-start"}`}
            >
              {m.role === "assistant" && (
                <div className="w-10 h-10 rounded-full bg-blue-500 text-white flex items-center justify-center">
                  🎨
                </div>
              )}

              <div
                className={`max-w-[75%] px-4 py-3 rounded-2xl shadow ${
                  m.role === "user"
                    ? "bg-gradient-to-r from-blue-600 to-blue-500 text-white"
                    : "bg-white"
                }`}
              >
                {m.content}
              </div>

              {m.role === "user" && (
                <div className="w-10 h-10 rounded-full bg-blue-600 text-white flex items-center justify-center">
                  👤
                </div>
              )}
            </div>
          );
        })}

        {askingCriterion && (
          <div className="bg-white rounded-xl border p-3 space-y-3">
            <p>어떤 기준으로 유사한 작품을 찾을까요?</p>
            <button
              onClick={() => handleSendMessage("시각적으로 유사한 작품")}
              className="w-full py-3 border rounded-lg"
            >
              시각적 유사도
            </button>
            <button
              onClick={() => handleSendMessage("내용/설명이 유사한 작품")}
              className="w-full py-3 border rounded-lg"
            >
              설명/내용 유사도
            </button>
          </div>
        )}

        {isTyping && <p className="text-blue-700 text-sm">AI 도슨트가 답변 중...</p>}

        <div ref={messagesEndRef} />
      </div>

      {/* 입력창 */}
      <div className="bg-white border-t p-4">
        {isSpeaking && (
          <div className="mb-3 bg-blue-50 border px-3 py-2 rounded flex justify-between">
            <span className="text-blue-600">AI 도슨트가 말하는 중...</span>
            <button onClick={stopSpeaking} className="text-blue-600 underline">
              중지
            </button>
          </div>
        )}

        <div className="flex gap-2 items-center">
          <button
            onClick={toggleMic}
            className={`p-3 rounded-lg border ${
              isListening ? "bg-red-50 border-red-300 text-red-600" : "bg-gray-50"
            }`}
          >
            {isListening ? <MicOff /> : <Mic />}
          </button>

          <input
          className="flex-1 px-4 py-3 border rounded-lg"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}

          // 🔥 한글 조합 시작
          onCompositionStart={() => setIsComposing(true)}

          // 🔥 한글 조합 끝
          onCompositionEnd={() => setIsComposing(false)}

          onKeyDown={(e) => {
            // 🔥 한글 조합 중에는 Enter 무조건 무시
            if (isComposing) return;

            // 🔥 Shift + Enter 는 줄바꿈 허용
            if (e.key === "Enter" && e.shiftKey) return;

            // 🔥 한글 자모로 끝나는 경우도 조합 중으로 간주 → 전송 방지
            const lastChar = inputValue.slice(-1);
            const isHangulJamo = /[ㄱ-ㅎㅏ-ㅣ]/.test(lastChar);
            if (isHangulJamo) return;

            // 🔥 Enter → 메시지 전송
            if (e.key === "Enter") {
              e.preventDefault();
              handleSendMessage();
            }
          }}

          placeholder="궁금한 점을 입력하세요"
        />


          <button
            onClick={() => handleSendMessage()}
            className="bg-blue-600 text-white px-4 py-3 rounded-lg disabled:opacity-40"
            disabled={!inputValue.trim()}
          >
            <Send />
          </button>
        </div>
      </div>

      {toastMsg && (
        <div className="fixed bottom-6 right-4 bg-black/80 text-white px-4 py-2 rounded-lg shadow">
          {toastMsg}
        </div>
      )}
    </div>
  );
}
