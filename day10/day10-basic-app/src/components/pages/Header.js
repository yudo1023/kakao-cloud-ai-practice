// ./components/pages/Header.js

import { UserContext } from "../../store/UserContext";
import { useContext } from "react";

const Header = () => {
    const name = useContext(UserContext);

    return (
        <header className="header">
            <p>
            <span>{name}</span>님 환영합니다.
            </p>
        </header>
    );
    
}

export default Header;