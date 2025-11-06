// ./components/example/UseContextDemo
// 화면 모드(라이트/다크 모드) 예제

import React, { useContext } from "react";
import { ThemeContext } from "../../store/ThemeContext";

const Theme = () => {
    const {theme, setTheme} = useContext(ThemeContext);

    return (
        <button onClick={() => setTheme(theme === "light" ? "dark" : "light")}>
            Current Theme: {theme}
        </button>
    );
};

export default Theme;