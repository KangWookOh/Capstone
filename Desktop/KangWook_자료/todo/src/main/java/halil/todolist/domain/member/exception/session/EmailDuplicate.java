package halil.todolist.domain.member.exception.session;

public class EmailDuplicate extends RuntimeException {

    private static final String MESSAGE = "이미 사용중인 이메일입니다.";

    public EmailDuplicate() {
        super(MESSAGE);
    }
}
