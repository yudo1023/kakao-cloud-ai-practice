// ./components/example/ProfileCard.js

function ProfileCard(props) {
    return (
        <div className="profile-card">
            <img src={props.avatarUrl} alt="Profile" />
            <h2>{props.name}</h2>
            <p>{props.bio}</p>
        </div>
    );
}

export default ProfileCard;