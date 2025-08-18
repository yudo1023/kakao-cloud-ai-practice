// ./components/example/Header.js

import { useContext } from "react";
import { UserContext } from "../../store/UserContext";
import { ThemeContext } from "../../store/ThemeContext";

const Main = () => {
    const [user, setUser] = useContext(UserContext);
    const {theme, setTheme} = useContext(ThemeContext);

    return (
        <header className={`header ${theme}`}>
            <p><span>{user.name}(나이: {user.age})</span>님 환영합니다.</p>
            <p>현재 테마는 {theme}입니다.</p>
            <button onClick={() => setTheme(theme === "light" ? "dark" : "light")}>테마 변경</button>
        </header>
    );
}
export default Main;