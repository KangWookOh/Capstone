package halil.todolist.domain.member.login.cookie;

import halil.todolist.domain.member.dto.SignUpDto;
import halil.todolist.domain.member.entity.Member;
import halil.todolist.domain.member.service.MemberService;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockHttpServletResponse;
import org.springframework.transaction.annotation.Transactional;

import javax.servlet.http.Cookie;

import static org.assertj.core.api.Assertions.assertThat;

@SpringBootTest
class CookieServiceTest {

    @Autowired
    CookieService cookieService;

    @Autowired
    MemberService memberService;

    @Test
    @Transactional
    @DisplayName("쿠키 조회")
    void getCookie() {
        // given
        MockHttpServletRequest request = new MockHttpServletRequest();
        MockHttpServletResponse response = new MockHttpServletResponse();

        Member member = Member.builder()
                .email("test@email.com")
                .password("1234").build();

        SignUpDto signUpDto = new SignUpDto();
        signUpDto.setEmail(member.getEmail());
        signUpDto.setPassword(member.getPassword());

        memberService.signUp(signUpDto);

        // when
        Member login = cookieService.login(response, signUpDto.getEmail(), signUpDto.getPassword());
        request.setCookies(response.getCookies());

        Cookie[] cookies = request.getCookies();            // 모든 쿠키 가져오기
        String findCookie = null;

        for (Cookie x : cookies) {
            String CookieName = x.getName();
            String CookieVal = x.getValue();
            System.out.println("쿠키 " + CookieVal);
            if (CookieVal.equals(String.valueOf(login.getId().intValue()))) {
                findCookie = CookieVal;
            }
        }
        System.out.println("파인드" + findCookie);

        // then
        assertThat(String.valueOf(login.getId().intValue())).isEqualTo(findCookie);
    }
}