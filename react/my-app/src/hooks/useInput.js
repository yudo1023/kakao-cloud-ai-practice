// ../hooks/useInput.js

import { useState } from "react";

export function useInput (initValue) {
    const [ inputValue, setInputValue ] = useState(initValue);

    const handleChange = (e) => {
        setInputValue(e.target.value);
    }

    const handleSubmit = () => {
        if (inputValue !== "") {
            window.alert("전송완료");
        }
    }

    return [ inputValue, handleChange, handleSubmit ];
}