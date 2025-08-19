// ./hooks/useProducts.js

import { useMemo, useCallback } from 'react';

const useProducts = (productCards) => {
  const cards = useMemo(() => productCards, [productCards]);
  const formatPrice = useCallback((num) => new Intl.NumberFormat('ko-KR').format(num), []);

  return { cards, formatPrice };
};

export default useProducts;