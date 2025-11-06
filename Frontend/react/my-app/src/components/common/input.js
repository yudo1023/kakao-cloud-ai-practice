//./components/common/input.js
import { useEffect, useState } from "react";
import { useInput } from "../../hooks/UseInput";

const Input = () => {
    // const [ inputValue, setInputValue ] = useState("");
    const [ inputValue, handleChange, handleSubmit ] = useInput(""); 
    // const [ inputValue01, handleChange01 ] = useInput(""); 
    // const [ inputValue02, handleChange02 ] = useInput(""); 
    // const [ inputValue03, handleChange03 ] = useInput(""); 

    // const handleChange = (e) => {
    //     setInputValue(e.target.value);
    // }

    // const handleSubmit = () => {
    //     if (inputValue !== "") {
    //         window.alert("전송완료");
    //     }
    // }

    useEffect(() => {
        console.log(inputValue);
    }, [inputValue]);

    return (
        <div>
            <input
                type="text"
                value={inputValue}
                onChange={handleChange}
                // 인라인 함수로 표기 가능
                // onChange={(e) => setInputvalue(e.target.value)}
                placeholder="텍스트를 입력하세요"
            />
            {/* <input
                type="text"
                value={inputValue01}
                onChange={handleChange01}
                placeholder="텍스트를 입력하세요"
            />
            <input
                type="text"
                value={inputValue02}
                onChange={handleChange02}
                placeholder="텍스트를 입력하세요"
            />
            <input
                type="text"
                value={inputValue03}
                onChange={handleChange03}
                placeholder="텍스트를 입력하세요"
            /> */}
            <button onClick={handleSubmit}>전송</button>
        </div>
    );
}

export default Input;