import { User, LogOut, Camera, History, Mail, GraduationCap, Users, Sparkles } from 'lucide-react';
import { DocentVersion } from '../App';

type ProfileScreenProps = {
  user: { email: string; name: string; docentVersion?: DocentVersion };
  onLogout: () => void;
  onVersionChange: (version: DocentVersion) => void;
};

const VERSION_INFO = {
  general: { title: '관람객 눈높이의 설명', icon: Users, color: 'text-blue-600', bg: 'bg-blue-100' },
  child: { title: '어린이용 버전', icon: Sparkles, color: 'text-pink-600', bg: 'bg-pink-100' },
  expert: { title: '전문가용 버전', icon: GraduationCap, color: 'text-emerald-600', bg: 'bg-emerald-100' },
};

export function ProfileScreen({ user, onLogout, onVersionChange }: ProfileScreenProps) {
  const handleLogout = () => {
    if (confirm('로그아웃 하시겠습니까?')) {
      onLogout();
    }
  };

  // 저장된 히스토리 가져오기
  const viewHistory = JSON.parse(localStorage.getItem(`history_${user.email}`) || '[]');

  const cameraCount = viewHistory.filter((h: any) => h.mode === "camera").length;
  const uniqueArtworksCount = new Set(viewHistory.map((h: any) => h.artworkId)).size;

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      <div className="bg-white shadow-sm p-4">
        <h1 className="text-center">내 계정</h1>
      </div>

      <div className="flex-1 overflow-y-auto p-4 pb-20">
        {/* Profile Card */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-4">
          <div className="flex items-center gap-4 mb-6">
            <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center">
              <User className="w-8 h-8 text-blue-600" />
            </div>

            <div className="flex-1">
              <h2>{user.name}</h2>
              <div className="flex items-center gap-2 text-gray-600 mt-1">
                <Mail className="w-4 h-4" />
                <p>{user.email}</p>
              </div>
            </div>
          </div>

          {/* 카운트 정보 */}
          <div className="grid grid-cols-2 gap-4 pt-6 border-t">
            {/* 📸 촬영 횟수 */}
            <div className="text-center">
              <div className="flex items-center justify-center gap-2 mb-1">
                <Camera className="w-5 h-5 text-blue-600" />
                <p className="text-blue-600">촬영 횟수</p>
              </div>
              <p>{cameraCount}회</p>
            </div>

            {/* 🖼 감상한 작품 수 */}
            <div className="text-center">
              <div className="flex items-center justify-center gap-2 mb-1">
                <History className="w-5 h-5 text-blue-600" />
                <p className="text-blue-600">감상한 작품</p>
              </div>
              <p>{uniqueArtworksCount}점</p>
            </div>
          </div>
        </div>

        {/* Docent Version Setting */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-4">
          <h3 className="mb-4">도슨트 설명 스타일</h3>

          <div className="space-y-3">
            {(Object.keys(VERSION_INFO) as DocentVersion[]).map((version) => {
              const info = VERSION_INFO[version];
              const Icon = info.icon;
              const isSelected = user.docentVersion === version;

              return (
                <button
                  key={version}
                  onClick={() => onVersionChange(version)}
                  className={`w-full flex items-center gap-3 p-4 rounded-lg border-2 transition-all ${
                    isSelected
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:border-blue-300 hover:bg-gray-50'
                  }`}
                >
                  <div className={`w-10 h-10 rounded-lg ${info.bg} flex items-center justify-center`}>
                    <Icon className={`w-5 h-5 ${info.color}`} />
                  </div>

                  <div className="flex-1 text-left">
                    <p className={isSelected ? 'text-blue-900' : 'text-gray-700'}>
                      {info.title}
                    </p>
                  </div>

                  {isSelected && (
                    <div className="w-6 h-6 rounded-full bg-blue-600 flex items-center justify-center text-white">
                      <span className="text-sm">✓</span>
                    </div>
                  )}
                </button>
              );
            })}
          </div>
        </div>

        {/* Viewing History */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-4">
          <h3 className="mb-4">최근 감상한 작품</h3>

          {viewHistory.length === 0 ? (
            <div className="text-center py-8">
              <Camera className="w-12 h-12 text-gray-300 mx-auto mb-2" />
              <p className="text-gray-500">아직 감상한 작품이 없습니다</p>
              <p className="text-gray-400 mt-1">작품을 촬영해보세요!</p>
            </div>
          ) : (
            <div className="space-y-3">
              {viewHistory
                .slice(0, 5)
                .reverse()
                .map((item: any, index: number) => (
                  <div key={index} className="flex gap-3 p-3 bg-gray-50 rounded-lg">
                    <img
                      src={item.imageUrl}
                      alt={item.title}
                      className="w-16 h-16 object-cover rounded"
                    />
                    <div className="flex-1">
                      <p className="line-clamp-1">{item.title}</p>
                      <p className="text-gray-600">{item.artist}</p>
                      <p className="text-gray-400 mt-1">
                        {new Date(item.timestamp).toLocaleDateString('ko-KR')}
                      </p>
                    </div>
                  </div>
                ))}
            </div>
          )}
        </div>

        {/* Logout Button */}
        <button
          onClick={handleLogout}
          className="w-full bg-white border border-red-200 text-red-600 py-3 px-4 rounded-lg hover:bg-red-50 transition-colors flex items-center justify-center gap-2"
        >
          <LogOut className="w-5 h-5" />
          로그아웃
        </button>
      </div>
    </div>
  );
}
