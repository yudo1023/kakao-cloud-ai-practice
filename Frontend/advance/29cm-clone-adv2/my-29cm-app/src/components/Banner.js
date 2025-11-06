// ./components/Banner.js (메인의 오른쪽 영역의 상위 배너)

import React from 'react';

const Banner = ({ alt = '', src }) => (
  <a className="tile2" aria-label={alt}>
    <img alt={alt} loading="lazy" decoding="async" src={src} />
  </a>
);

export default React.memo(Banner);
