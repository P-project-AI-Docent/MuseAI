import { Camera, MessageCircle, Search, Mic, Volume2 } from 'lucide-react';

type WelcomeScreenProps = {
  userName: string;
  onStart: () => void;
};

export function WelcomeScreen({ userName, onStart }: WelcomeScreenProps) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-cyan-50 to-teal-50 flex flex-col">
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-2xl mx-auto px-6 py-8">
          {/* Header */}
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-24 h-24 rounded-full bg-gradient-to-br from-blue-500 to-cyan-500 text-white shadow-lg mb-4">
              <span className="text-5xl">🎨</span>
            </div>
            <h1 className="mb-2 bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent">
              AI 도슨트에 오신 것을 환영합니다!
            </h1>
            <p className="text-gray-600">{userName}님, 함께 예술 여행을 시작해볼까요?</p>
          </div>

          {/* Features */}
          <div className="space-y-6 mb-8">
            <div className="bg-white rounded-2xl p-6 shadow-md hover:shadow-lg transition-shadow">
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-12 h-12 rounded-full bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center text-white shadow-md">
                  <Camera className="w-6 h-6" />
                </div>
                <div className="flex-1">
                  <h3 className="mb-2 text-blue-900">1. 작품 촬영하기</h3>
                  <p className="text-gray-600 leading-relaxed">
                    미술관에서 마음에 드는 작품을 카메라로 촬영하거나 갤러리에서 이미지를 선택하세요.
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-2xl p-6 shadow-md hover:shadow-lg transition-shadow">
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-12 h-12 rounded-full bg-gradient-to-br from-cyan-500 to-cyan-600 flex items-center justify-center text-white shadow-md">
                  <Search className="w-6 h-6" />
                </div>
                <div className="flex-1">
                  <h3 className="mb-2 text-cyan-900">2. 작품 확인하기</h3>
                  <p className="text-gray-600 leading-relaxed">
                    AI가 작품을 자동으로 인식합니다. 매칭이 정확하지 않다면 작품명으로 검색할 수 있어요.
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-2xl p-6 shadow-md hover:shadow-lg transition-shadow">
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-12 h-12 rounded-full bg-gradient-to-br from-teal-500 to-teal-600 flex items-center justify-center text-white shadow-md">
                  <MessageCircle className="w-6 h-6" />
                </div>
                <div className="flex-1">
                  <h3 className="mb-2 text-teal-900">3. AI 도슨트와 대화하기</h3>
                  <p className="text-gray-600 leading-relaxed">
                    작품의 역사, 기법, 의미 등 궁금한 점을 자유롭게 질문하세요. AI 도슨트가 친절하게 답변해드립니다.
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-gradient-to-r from-blue-100 to-cyan-100 rounded-2xl p-6 shadow-md border-2 border-blue-200">
              <h3 className="mb-3 text-blue-900 flex items-center gap-2">
                <span>✨</span>
                특별 기능
              </h3>
              <div className="space-y-2 text-gray-700">
                <div className="flex items-center gap-2">
                  <Mic className="w-5 h-5 text-blue-600" />
                  <span>음성으로 질문하기</span>
                </div>
                <div className="flex items-center gap-2">
                  <Volume2 className="w-5 h-5 text-blue-600" />
                  <span>답변을 음성으로 듣기</span>
                </div>
              </div>
            </div>
          </div>

          {/* Tips */}
          <div className="bg-blue-50 rounded-2xl p-6 mb-6 border border-blue-200">
            <h3 className="mb-3 text-blue-900">💡 이용 팁</h3>
            <ul className="space-y-2 text-gray-700">
              <li className="flex items-start gap-2">
                <span className="text-blue-600 mt-1">•</span>
                <span>작품 촬영 시 조명이 밝고 작품이 선명하게 보이도록 해주세요.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-600 mt-1">•</span>
                <span>카메라 권한이 거부되면 갤러리에서 이미지를 선택할 수 있습니다.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-600 mt-1">•</span>
                <span>개인 페이지에서 감상 기록을 확인할 수 있습니다.</span>
              </li>
            </ul>
          </div>
        </div>
      </div>

      {/* Start Button */}
      <div className="bg-white/80 backdrop-blur-sm border-t border-blue-100 p-6 safe-area-bottom shadow-lg">
        <button
          onClick={onStart}
          className="w-full max-w-2xl mx-auto bg-gradient-to-r from-blue-600 to-cyan-600 text-white py-4 rounded-xl hover:from-blue-700 hover:to-cyan-700 transition-all shadow-lg shadow-blue-300 flex items-center justify-center gap-2"
        >
          <Camera className="w-6 h-6" />
          <span className="text-lg">시작하기</span>
        </button>
      </div>
    </div>
  );
}
