// ./components/example/CounterClass.js

import React, {Component} from "react";

class CounterClass extends Component {
    constructor(props) {
        super(props);
        this.state = {
            number: 0,
        };
    }

    componentDidMount() {
        console.log("ComponentDidMount(클래스형)");
    }

    componentDidUpdate(prevProps, prevState) {
        console.log("ComponenetDidUpdate(클래스형)");
    }

    componentWillUnmount() {
        console.log("componentWillUnmount(클래스형)")
    }

    handleIncrease = () => {
        this.setState({
            number: this.state.number + 1,
        });
    };

    handleDecrease = () => {
        this.setState({
            number: this.state.number - 1,
        });
    };

    render() {
        console.log("render(클래스형)");
        return (
            <div>
                <h1 style={{ color: "green" }}>클래스형 컴포넌트</h1>
                <p>Count: {this.state.number}</p>
                <button onClick={this.handleIncrease}>증가</button>
                <button onClick={() => this.handleDecrease()}>감소</button>
            </div>
        );
    }
}

export default CounterClass;