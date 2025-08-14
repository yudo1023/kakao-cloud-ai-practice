// ./components/example/UseStateDemo.js

import { useState } from 'react';

const UseStateDemo = () => {
    // const [ state, setState ] = useState("");

    // const [ userData, setuserData ] = useState("");
    // const [ tableData, settableData ] = useState({});

    // const [data, setData]  = useState({
    //     user: "",
    //     table: []
    // }) 

    // 상태가 바뀌지 않음
    // let count = 0;
    // const handleClick = ( ) => {
    //     count = count + 1;
    //     console.log(count);
    // }

    // const [ count, setCount ] = useState(0);

    // const handleClick = ( ) => {
    //     setCount((count) => count+1);
    //     // setState(currentState); prevState => CurrentState
    // }

    // const [ arrData, setArrData ] = useState([1, 2, 3]);
    
    // const handleClick = () => {
    //     // 잘못된 방법
    //     // arrData.push(4);
    //     // setArrData(arrData);

    //     setArrData([...arrData, 4]);
    // }

    const [ userInput, setUserInput ] = useState({
        title: "",
        content: "",
        date: ""
    });

    const handleClick = () => {

    };

    const handleChange = (e) => {
        // userInput.title = e.target.value;

        // setUserInput({
        //     ...userInput,
        //     title: e.target.value
        // });

        setUserInput((prevState) => {
            return { ...prevState, title: e.target.value };
        });
    };

    return (
        <div>
            {/* <p>현재 카운트 : {count}</p> */}
            <p>타이틀 : {userInput.title} </p>
            <input
                value={userInput.title}
                onChange={handleChange}
            />
            {/* <button onClick={handleClick}>증가</button> */}
        </div>
    );
};

export default UseStateDemo;