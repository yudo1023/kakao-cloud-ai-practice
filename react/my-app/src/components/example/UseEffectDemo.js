// ./components/example/UseEffectDemo.js

import { useState, useEffect } from "react";

const UseEffectDemo = () => {
    const [ count, setCount ] = useState(0);
    const [ userInput, setUserInput ] = useState({
            title: "",
            content: "",
            date: ""
        });

    // 의존성 배열 안에 값이 변함에 따라 동작하는 함수
    useEffect(() => {
        console.log("ComponentDidMount");
        // 초기 API 데이터 Fetching
    }, []);

    useEffect(() => {
        console.log("ComponentDidMount + ComponentDidUpdate");
        // 특별한 의미 x, 계속 반복해서 호출됨
    });

    useEffect(() => {
        console.log("ComponentDidMount + ComponentDidUpdate(Count만 변경시)");
         // 특정 상태값 변화에 따른 API Fetching
    }, [count]);

    useEffect(() => {
        console.log("ComponentDidMount + ComponentDidUpdate(Input만 변경시)");
    }, [userInput]);

    useEffect(() => {
        console.log("ComponentDidMount + ComponentDidUpdate(Title 변경시)");
    }, [userInput.title]);

    useEffect(() => {
        console.log("ComponentDidMount + ComponentDidUpdate(Title 변경시)");
        // 여러개 가능
    }, [count, userInput.title]);


    const handleClick = () => {
        setCount(count+1);
    };

    const handleChange = (e) => {
        setUserInput((prevState) => {
            return { ...prevState, title: e.target.value };
        });
    }

    return (
        <div>
            <p>카운트 : {count} </p>
            <button onClick={handleClick}>증가</button> 
            <input
                value={userInput.title}
                onChange={handleChange}
            />
        </div>
    );
};

export default UseEffectDemo;