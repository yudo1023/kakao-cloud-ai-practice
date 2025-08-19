// ./hooks/useSubmenu.js

import { useCallback, useState } from 'react';

const useSubmenu = () => {
  const [activeMenu, setActiveMenu] = useState(null);

  const toggleMenu = useCallback((menu) => {
    setActiveMenu(prev => (prev === menu ? null : menu));
  }, []);

  const closeMenu = useCallback(() => {
    setActiveMenu(null);
  }, []);

  return {
    activeMenu,
    isOpen: Boolean(activeMenu),
    toggleMenu,
    closeMenu,
  };
};

export default useSubmenu;
