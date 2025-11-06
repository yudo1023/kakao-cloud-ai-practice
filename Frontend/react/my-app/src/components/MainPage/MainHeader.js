// ./components/MainPage/MainHeader.js

import Button from "../common/buttons"

/*
function MainHeader(props) {
    return (
        <header>
          <h1>안녕하세요. {props.name}님! </h1>
          <Button userName={props.name}/>
        </header>
      );
}
*/

function MainHeader({userName}) {
    return (
        <header>
          <h1>안녕하세요. {userName}님! </h1>
          <Button userName={userName}/>
        </header>
      );
}

export default MainHeader;