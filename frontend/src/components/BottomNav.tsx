import { Camera, User } from "lucide-react";

type BottomNavProps = {
  currentPage: "camera" | "profile";
  onNavigate: (page: "camera" | "profile") => void;
};

export function BottomNav({ currentPage, onNavigate }: BottomNavProps) {
  return (
    <div className="fixed bottom-0 left-0 right-0 bg-white border-t border-gray-200">
      {/* Safe area padding for iPhone */}
      <div className="pb-[env(safe-area-inset-bottom)]">
        <div className="flex">
          {/* 촬영 버튼 */}
          <button
            onClick={() => onNavigate("camera")}
            className={`flex-1 flex flex-col items-center gap-1 py-3 transition-colors ${
              currentPage === "camera"
                ? "text-blue-600"
                : "text-gray-500 hover:text-gray-700"
            }`}
          >
            <Camera className="w-6 h-6" strokeWidth={2} />
            <span className="text-[11px] font-medium">촬영</span>
          </button>

          {/* 내 계정 버튼 */}
          <button
            onClick={() => onNavigate("profile")}
            className={`flex-1 flex flex-col items-center gap-1 py-3 transition-colors ${
              currentPage === "profile"
                ? "text-blue-600"
                : "text-gray-500 hover:text-gray-700"
            }`}
          >
            <User className="w-6 h-6" strokeWidth={2} />
            <span className="text-[11px] font-medium">내 계정</span>
          </button>
        </div>
      </div>
    </div>
  );
}
