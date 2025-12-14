import { useState } from 'react';
import { Search, ArrowLeft, Camera } from 'lucide-react';
import { Artwork } from '../App';

type ArtworkSearchProps = {
  onSelect: (artwork: Artwork) => void;
  onBackToCamera: () => void;
};

const API_BASE = "https://localhost:8001";

export function ArtworkSearch({ onSelect, onBackToCamera }: ArtworkSearchProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<Artwork[]>([]);
  const [hasSearched, setHasSearched] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;

    try {
      setLoading(true);
      setHasSearched(true);
      setSearchResults([]);

      const r = await fetch(`${API_BASE}/api/text/search?q=${encodeURIComponent(searchQuery)}&limit=30`);
      if (!r.ok) throw new Error("검색 실패");

      const { results } = await r.json();

      const metas: Artwork[] = await Promise.all(
        (results || []).map(async (row: any) => {
          const r2 = await fetch(`${API_BASE}/api/artwork/${row.objectID}`);

          if (!r2.ok) {
            return {
              id: String(row.objectID),
              title: row.title || "",
              artist: row.artist || "",
              year: "",
              description: "",
              imageUrl: `${API_BASE}/images/${row.objectID}.jpg`,
            } as Artwork;
          }

          const meta = await r2.json();

          return {
            id: meta.id,
            title: meta.title || row.title || "",
            artist: meta.artist || row.artist || "",
            year: meta.year || "",
            description: meta.description || "",
            imageUrl: `${API_BASE}/images/${meta.id}.jpg`,
          } as Artwork;
        })
      );

      setSearchResults(metas);
    } catch (e) {
      console.error(e);
      setSearchResults([]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleSearch();
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      <div className="bg-white shadow-sm p-4">
        <div className="flex items-center gap-3 mb-3">
          <button onClick={onBackToCamera} className="p-2 hover:bg-gray-100 rounded-lg transition-colors">
            <ArrowLeft className="w-5 h-5" />
          </button>
          <h1 className="flex-1 text-center">작품명 검색</h1>
          <div className="w-9" />
        </div>
        <p className="text-center text-gray-600">작품명 또는 작가명으로 검색하세요</p>
      </div>

      <div className="p-4">
        <div className="flex gap-2">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="예: 별이 빛나는 밤, 반 고흐"
            className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600 focus:border-transparent"
          />
          <button
            onClick={handleSearch}
            className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2"
          >
            <Search className="w-5 h-5" /> 검색
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        {!hasSearched ? (
          <div className="text-center py-12">
            <Search className="w-16 h-16 text-gray-300 mx-auto mb-4" />
            <p className="text-gray-500">작품명이나 작가명을 입력해주세요</p>
            <div className="mt-8 space-y-2 text-left max-w-md mx-auto bg-blue-50 p-4 rounded-lg">
              <p className="text-blue-900">💡 검색 팁</p>
              <ul className="list-disc list-inside text-blue-700 space-y-1 ml-2">
                <li>작품 제목의 일부만 입력해도 됩니다</li>
                <li>작가 이름으로도 검색 가능합니다</li>
                <li>띄어쓰기에 주의해주세요</li>
              </ul>
            </div>
          </div>
        ) : loading ? (
          <div className="text-center py-12 text-gray-600">검색 중...</div>
        ) : searchResults.length === 0 ? (
          <div className="text-center py-12">
            <Search className="w-16 h-16 text-gray-300 mx-auto mb-4" />
            <p className="text-gray-700 mb-2">검색 결과가 없습니다</p>
            <p className="text-gray-500">다른 키워드로 검색해보세요</p>
            <button
              onClick={onBackToCamera}
              className="mt-6 flex items-center gap-2 mx-auto text-blue-600 hover:text-blue-700"
            >
              <Camera className="w-5 h-5" /> 다시 촬영하기
            </button>
          </div>
        ) : (
          <div className="space-y-4 max-w-2xl mx-auto">
            <p className="text-gray-600">{searchResults.length}개의 작품을 찾았습니다</p>
            {searchResults.map((artwork) => (
              <div
                key={artwork.id}
                className="bg-white rounded-lg shadow-sm overflow-hidden hover:shadow-md transition-shadow cursor-pointer"
                onClick={() => onSelect(artwork)}
              >
                <div className="flex gap-4 p-4">
                  {artwork.imageUrl ? (
                    <img
                      src={artwork.imageUrl}
                      alt={artwork.title}
                      className="w-24 h-24 object-cover rounded flex-shrink-0"
                    />
                  ) : (
                    <div className="w-24 h-24 rounded bg-gray-200 flex items-center justify-center text-gray-500">
                      이미지 없음
                    </div>
                  )}
                  <div className="flex-1 min-w-0">
                    <h3 className="mb-1 line-clamp-1">{artwork.title}</h3>
                    <p className="text-gray-600 mb-2">
                      {artwork.artist}
                      {artwork.year ? ` · ${artwork.year}` : ""}
                    </p>
                    <p className="text-gray-500 line-clamp-2">
                      {artwork.description || "설명이 없습니다"}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
