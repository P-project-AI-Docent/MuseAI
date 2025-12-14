import { useState } from 'react';
import { GraduationCap, Users, Sparkles } from 'lucide-react';

type DocentVersion = 'general' | 'child' | 'expert';

type DocentVersionSelectorProps = {
  userName: string;
  onSelect: (version: DocentVersion) => void;
};

const VERSIONS = [
  {
    id: 'general' as DocentVersion,
    title: '관람객 눈높이의 설명',
    description: '일반적인 수준에서 이해하기 쉽게 설명합니다',
    icon: Users,
    color: 'from-blue-500 to-cyan-500',
    bgColor: 'bg-blue-50',
    borderColor: 'border-blue-200',
  },
  {
    id: 'child' as DocentVersion,
    title: '어린이용 버전',
    description: '재미있고 쉬운 언어로 어린이도 이해할 수 있게 설명합니다',
    icon: Sparkles,
    color: 'from-pink-500 to-purple-500',
    bgColor: 'bg-pink-50',
    borderColor: 'border-pink-200',
  },
  {
    id: 'expert' as DocentVersion,
    title: '전문가용 버전',
    description: '미술사적 배경과 전문적인 분석을 제공합니다',
    icon: GraduationCap,
    color: 'from-emerald-500 to-teal-500',
    bgColor: 'bg-emerald-50',
    borderColor: 'border-emerald-200',
  },
];

export function DocentVersionSelector({ userName, onSelect }: DocentVersionSelectorProps) {
  const [selectedVersion, setSelectedVersion] = useState<DocentVersion | null>(null);

  const handleSelect = (version: DocentVersion) => {
    setSelectedVersion(version);
  };

  const handleConfirm = () => {
    if (selectedVersion) {
      onSelect(selectedVersion);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-cyan-50 to-teal-50 flex items-center justify-center p-4">
      <div className="max-w-2xl w-full">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gradient-to-br from-blue-600 to-cyan-600 text-white shadow-lg mb-4">
            <span className="text-2xl">🎨</span>
          </div>
          <h1 className="mb-2 text-blue-900">AI 도슨트 설정</h1>
          <p className="text-gray-600">
            {userName}님, 환영합니다!<br />
            원하시는 도슨트 설명 스타일을 선택해주세요
          </p>
        </div>

        <div className="space-y-4 mb-6">
          {VERSIONS.map((version) => {
            const Icon = version.icon;
            const isSelected = selectedVersion === version.id;
            
            return (
              <button
                key={version.id}
                onClick={() => handleSelect(version.id)}
                className={`w-full text-left p-6 rounded-2xl border-2 transition-all ${
                  isSelected
                    ? `${version.bgColor} ${version.borderColor} shadow-lg scale-105`
                    : 'bg-white border-gray-200 hover:border-blue-300 hover:shadow-md'
                }`}
              >
                <div className="flex items-start gap-4">
                  <div
                    className={`flex-shrink-0 w-12 h-12 rounded-xl bg-gradient-to-br ${version.color} flex items-center justify-center text-white shadow-md`}
                  >
                    <Icon className="w-6 h-6" />
                  </div>
                  <div className="flex-1">
                    <h3 className="mb-1 text-gray-900">{version.title}</h3>
                    <p className="text-gray-600 text-sm leading-relaxed">
                      {version.description}
                    </p>
                  </div>
                  {isSelected && (
                    <div className="flex-shrink-0">
                      <div className="w-6 h-6 rounded-full bg-gradient-to-br from-blue-600 to-cyan-600 flex items-center justify-center text-white">
                        <span className="text-sm">✓</span>
                      </div>
                    </div>
                  )}
                </div>
              </button>
            );
          })}
        </div>

        <button
          onClick={handleConfirm}
          disabled={!selectedVersion}
          className="w-full bg-gradient-to-r from-blue-600 to-cyan-600 text-white py-4 px-6 rounded-xl hover:from-blue-700 hover:to-cyan-700 transition-all shadow-lg disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:from-blue-600 disabled:hover:to-cyan-600"
        >
          {selectedVersion ? '선택 완료' : '도슨트 스타일을 선택해주세요'}
        </button>

        <p className="text-center text-gray-500 text-sm mt-4">
          나중에 내 계정에서 변경할 수 있습니다
        </p>
      </div>
    </div>
  );
}
