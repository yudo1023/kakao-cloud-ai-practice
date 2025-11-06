// ./components/Tile.js (메인의 왼쪽 영역의 이미지 타일)

import React from 'react';

const Tile = ({ alt, src }) => (
  <a className="tile" aria-label={alt}>
    <img alt={alt} loading="lazy" decoding="async" src={src} />
  </a>
);

export default React.memo(Tile);
